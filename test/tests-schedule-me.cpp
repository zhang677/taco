//
// Created by 张 on 2022/2/27.
//
#include <taco/index_notation/transformations.h>
#include <codegen/codegen_c.h>
#include <codegen/codegen_cuda.h>
#include <fstream>
#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/transformations.h"
#include "codegen/codegen.h"
#include "taco/lower/lower.h"
#include "op_factory.h"

using namespace taco;
namespace mytest {
    const IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
    int NUM_I = 256;//256;
    int NUM_J = 512;//512;
    int NUM_K = 128;//128;
    float SPARSITY = .3;
    Tensor<float> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<float> B("B", {NUM_J, NUM_K}, Format({{Dense, Dense},
                                                 {0,     1}}));
//Tensor<float> C("C", {NUM_I, NUM_K}, Format({{Dense, Dense}, {1, 0}}));//column major
    Tensor<float> C("C", {NUM_I, NUM_K}, Format({{Dense, Dense},
                                                 {0,     1}})); //row-major

    void _printToCout(IndexStmt stmt) {
        std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
        ir::Stmt compute = lower(stmt, "compute", false, true);
        codegen->compile(compute, true);
    }

    void _printToFile(const string& filename, const IndexStmt& stmt) {
        stringstream source;

        string file_path = "eval_generated/";
        mkdir(file_path.c_str(), 0777);

        std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
        ir::Stmt compute = lower(stmt, "compute",  false, true);
        codegen->compile(compute, true);

        ofstream source_file;
        string file_ending=".cu";
        source_file.open(file_path + filename + file_ending);
        source_file << source.str();
        source_file.close();
    }

    void _printIRtoFile(const string& filename, const IndexStmt& stmt) {
        stringstream source;
        string file_path = "eval_generated/";
        mkdir(file_path.c_str(), 0777);
        std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
        ir::Stmt compute = lower(stmt, "compute",  false, true);

        ofstream source_file;
        string file_ending=".txt";
        source_file.open(file_path + filename + file_ending);
        ir::IRPrinter irp = ir::IRPrinter(source_file);
        source_file<<stmt<<endl;
        irp.print(compute);
        source_file<<endl;

    }

    IndexStmt _prepare(Tensor<float>& A, Tensor<float>& B, Tensor<float>& C) {
        srand(434321);
        for (int i = 0; i < NUM_I; i++) {
            for (int j = 0; j < NUM_J; j++) {
                float rand_float = (float) rand() / (float) (RAND_MAX);
                if (rand_float < SPARSITY) {
                    A.insert({i, j}, (float) ((int) (rand_float * 3 / SPARSITY)));
                }
            }
        }

        for (int j = 0; j < NUM_J; j++) {
            for (int k = 0; k < NUM_K; k++) {
                float rand_float = (float) rand() / (float) (RAND_MAX);
                B.insert({j, k}, (float) ((int) (rand_float * 3 / SPARSITY)));
            }
        }
        A.pack();
        B.pack();
        C(i, k) = A(i, j) * B(j, k);
        IndexStmt stmt = C.getAssignment().concretize();
        return stmt;
    }

    IndexStmt SpMM_SR_RB(IndexStmt stmt, Tensor<float> A) {
        IndexVar block("block"), io("io"), warp("warp"), thread("thread");
        IndexVar warp_row("warp_row"), thread_col("thread_col");
        return stmt.reorder({i, k, j})
                .split(i, block, io, 16)
                .reorder({block, io, k, j})
                .split(io, warp, warp_row, 1)
                .reorder({block, warp, warp_row, k, j})
                .split(k, thread, thread_col, 4)
                .reorder({block, warp, warp_row, thread, thread_col, j})
                .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::NoRaces)
                .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::NoRaces)
                .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::NoRaces);
    }

    IndexStmt SpMM_SR_EB(IndexStmt stmt) {
        stmt = BypassOptimizeSpMM(stmt,0);
        return stmt;
    }

    IndexStmt SpMM_SR_EB_N(IndexStmt stmt) {
        stmt = BypassOptimizeSpMM(stmt, 1);
        return stmt;
    }

    IndexStmt SpMM_PR_RB(IndexStmt stmt, Tensor<float> A) {
        IndexVar io("io"), ko("ko"), ki("ki"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
        return scalarPromote(stmt.reorder({i, k, j})
                                     .fuse(i, k, io)
                                     .split(io, ko, ki, 8) // 32 warp thread per block
                                     .reorder({ko, ki, j})
                                     .pos(j, jpos, A(i, j))
                                     .reorder({ko, ki, jpos})
                                     .split(jpos, jpos0, jpos1, 32)
                                     .reorder({ko, ki, jpos1, jpos0})
                                     .parallelize(ko, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
                                     .parallelize(ki, ParallelUnit::GPUWarp, OutputRaceStrategy::Atomics) //
                                     .parallelize(jpos1, ParallelUnit::GPUThread,
                                                  OutputRaceStrategy::ParallelReduction));
    }

    IndexStmt scheduleSpGEMMCPU(IndexStmt stmt, bool doPrecompute) {
        Assignment assign = stmt.as<Forall>().getStmt().as<Forall>().getStmt()
                .as<Forall>().getStmt().as<Assignment>();
        TensorVar result = assign.getLhs().getTensorVar();

        stmt = reorderLoopsTopologically(stmt);
        if (doPrecompute) {
            IndexVar j = assign.getLhs().getIndexVars()[1];
            TensorVar w("w", Type(result.getType().getDataType(),
                                  {result.getType().getShape().getDimension(1)}), taco::dense);
            stmt = stmt.precompute(assign.getRhs(), j, j, w);
        }
        stmt = stmt.assemble(result, AssembleStrategy::Insert, true);
        auto qi_stmt = stmt.as<Assemble>().getQueries();
        IndexVar qi;
        if (isa<Where>(qi_stmt)) {
            qi = qi_stmt.as<Where>().getConsumer().as<Forall>().getIndexVar();
        } else {
            qi = qi_stmt.as<Forall>().getIndexVar();
        }
        stmt = stmt.parallelize(i, ParallelUnit::CPUThread,
                                OutputRaceStrategy::NoRaces)
                .parallelize(qi, ParallelUnit::CPUThread,
                             OutputRaceStrategy::NoRaces);

        return stmt;
    }

    IndexStmt prepareSpGEMM(Format aFormat, Format bFormat, bool doPrecompute){
        int NUM_I = 100;
        int NUM_J = 100;
        int NUM_K = 100;
        float SPARSITY = .03;
        Tensor<double> A("A", {NUM_I, NUM_J}, aFormat);
        Tensor<double> B("B", {NUM_J, NUM_K}, bFormat);
        Tensor<double> C("C", {NUM_I, NUM_K}, CSR);

        srand(75883);
        for (int i = 0; i < NUM_I; i++) {
            for (int j = 0; j < NUM_J; j++) {
                float rand_float = (float)rand()/(float)(RAND_MAX);
                if (rand_float < SPARSITY) {
                    A.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
                }
            }
        }

        for (int j = 0; j < NUM_J; j++) {
            for (int k = 0; k < NUM_K; k++) {
                float rand_float = (float)rand()/(float)(RAND_MAX);
                if (rand_float < SPARSITY) {
                    B.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
                }
            }
        }

        A.pack();
        B.pack();

        C(i, k) = A(i, j) * B(j, k);
        IndexStmt stmt = C.getAssignment().concretize();
        return scheduleSpGEMMCPU(stmt, doPrecompute);
    }

    TEST(scheduling_eval, SpGEMM) {
        IndexStmt stmt = prepareSpGEMM(DCSR, CSR, true);
        string filename = "spgemm_test";
        set_CUDA_codegen_enabled(0);
        _printToFile(filename,stmt);
        _printIRtoFile(filename,stmt);
        cout<<stmt<<endl;
        ASSERT_EQ(1, 1);
    }

    TEST(scheduling_eval, spmm_SR_EB_N) {
        IndexStmt stmt = _prepare(A,B,C);
        stmt = SpMM_SR_EB_N(stmt);
        //stmt = scalarPromote(stmt);
        string filename = "sr_eb_n_test";
        set_CUDA_codegen_enabled(1);
        _printToFile(filename,stmt);
        _printIRtoFile(filename,stmt);
        cout<<stmt<<endl;
        ASSERT_EQ(1, 1);
    }

    TEST(scheduling_eval, spmm_SR_RB) {
        IndexStmt stmt = _prepare(A,B,C);
        stmt = SpMM_SR_RB(stmt, A);
        stmt = scalarPromote(stmt);
        string filename = "sr_rb_test";
        set_CUDA_codegen_enabled(1);
        //_printToCout(stmt);
        _printToFile(filename,stmt);
        ASSERT_EQ(1, 1);
    }

    TEST(scheduling_eval, spmm_SR_EB) {

        IndexStmt stmt = _prepare(A,B,C);
        stmt = SpMM_SR_EB(stmt);
        //stmt = scalarPromote(stmt);
        string filename = "sr_eb_test";
        set_CUDA_codegen_enabled(1);
        _printToFile(filename,stmt);
        _printIRtoFile(filename,stmt);
        cout<<stmt<<endl;
        ASSERT_EQ(1, 1);
    }

    TEST(scheduling_eval, spmm_PR_RB) {

        IndexStmt stmt = _prepare(A,B,C);
        stmt = SpMM_PR_RB(stmt, A);
        stmt = scalarPromote(stmt);
        string filename = "pr_rb_test";
        set_CUDA_codegen_enabled(1);
        _printToCout(stmt);
        ASSERT_EQ(1, 1);
    }
}
