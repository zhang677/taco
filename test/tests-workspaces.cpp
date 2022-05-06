#include <taco/index_notation/transformations.h>
#include <codegen/codegen_c.h>
#include <codegen/codegen_cuda.h>
#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "codegen/codegen.h"
#include "taco/lower/lower.h"
#include "fstream"
using namespace taco;

namespace Temptest {
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
    void _printToCout(IndexStmt stmt) {
        std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
        ir::Stmt compute = lower(stmt, "compute", false, true);
        ir::IRPrinter irp = ir::IRPrinter(cout);
        irp.print(compute);
    }
    void _printToFile(string filename, IndexStmt stmt) {
        stringstream source;

        string file_path = "eval_generated/";
        mkdir(file_path.c_str(), 0777);

        std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
        ir::Stmt compute = lower(stmt, "compute",  true, true);
        codegen->compile(compute, true);

        ofstream source_file;
        string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
        source_file.open(file_path + filename + file_ending);
        source_file << source.str();
        source_file.close();
    }

    TEST(workspaces, tile_vecElemMul_NoTail) {

        Tensor<double> A("A", {16}, Format{Dense});
        Tensor<double> B("B", {16}, Format{Dense});
        Tensor<double> C("C", {16}, Format{Dense});

        for (int i = 0; i < 16; i++) {
            A.insert({i}, (double) i);
            B.insert({i}, (double) i);
        }

        A.pack();
        B.pack();

        IndexVar i("i");
        IndexVar i_bounded("i_bounded");
        IndexVar i0("i0"), i1("i1");
        IndexExpr precomputedExpr = B(i) * C(i);
        A(i) = precomputedExpr;

        IndexStmt stmt = A.getAssignment().concretize();
        TensorVar precomputed("precomputed", Type(Float64, {Dimension(i1)}), taco::dense);
        stmt = stmt.bound(i, i_bounded, 16, BoundType::MaxExact)
                .split(i_bounded, i0, i1, 4)
                .precompute(precomputedExpr, i1, i1, precomputed);

        A.compile(stmt);
        A.assemble();
        A.compute();
        //_printToCout(stmt);
        Tensor<double> expected("expected", {16}, Format{Dense});
        expected(i) = B(i) * C(i);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);
    }

    TEST(workspaces, tile_vecElemMul_Tail1) {

        Tensor<double> A("A", {16}, Format{Dense});
        Tensor<double> B("B", {16}, Format{Dense});
        Tensor<double> C("C", {16}, Format{Dense});

        for (int i = 0; i < 16; i++) {
            A.insert({i}, (double) i);
            B.insert({i}, (double) i);
        }

        A.pack();
        B.pack();

        IndexVar i("i");
        IndexVar i_bounded("i_bounded");
        IndexVar i0("i0"), i1("i1");
        IndexExpr precomputedExpr = B(i) * C(i);
        A(i) = precomputedExpr;

        IndexStmt stmt = A.getAssignment().concretize();
        TensorVar precomputed("precomputed", Type(Float64, {Dimension(i1)}), taco::dense);
        stmt = stmt.bound(i, i_bounded, 16, BoundType::MaxExact)
                .split(i_bounded, i0, i1, 5)
                .precompute(precomputedExpr, i1, i1, precomputed);

        A.compile(stmt.concretize());
        A.assemble();
        A.compute();
        //_printToCout(stmt);
        Tensor<double> expected("expected", {16}, Format{Dense});
        expected(i) = B(i) * C(i);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);
    }

    TEST(workspaces, tile_vecElemMul_Tail2) {

        Tensor<double> A("A", {17}, Format{Dense});
        Tensor<double> B("B", {17}, Format{Dense});
        Tensor<double> C("C", {17}, Format{Dense});

        for (int i = 0; i < 17; i++) {
            A.insert({i}, (double) i);
            B.insert({i}, (double) i);
        }

        A.pack();
        B.pack();

        IndexVar i("i");
        IndexVar i_bounded("i_bounded");
        IndexVar i0("i0"), i1("i1");
        IndexExpr precomputedExpr = B(i) * C(i);
        A(i) = precomputedExpr;

        IndexStmt stmt = A.getAssignment().concretize();
        TensorVar precomputed("precomputed", Type(Float64, {Dimension(i1)}), taco::dense);
        stmt = stmt//.bound(i, i_bounded, 17, BoundType::MaxExact)
                //.split(i_bounded, i0, i1, 4)
                .split(i, i0, i1, 4)
                .precompute(precomputedExpr, i1, i1, precomputed);

        A.compile(stmt.concretize());
        A.assemble();
        A.compute();
        //_printToCout(stmt);
        Tensor<double> expected("expected", {17}, Format{Dense});
        expected(i) = B(i) * C(i);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);

//  ir::IRPrinter irp = ir::IRPrinter(cout);
//    
//  cout << stmt << endl;
//
//  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
//  ir::Stmt compute = lower(stmt, "compute",  false, true);
//  
//  irp.print(compute);
//  cout << endl;
//  codegen->compile(compute, false);
    }

    TEST(workspaces, tile_denseVecMul) {

        Tensor<double> A("A", {16}, Format{Dense});
        Tensor<double> B("B", {16}, Format{Dense});
        Tensor<double> C("C", {16}, Format{Dense});

        for (int i = 0; i < 16; i++) {
            B.insert({i}, (double) i);
            C.insert({i}, (double) i);
        }

        A.pack();
        B.pack();

        IndexVar i("i");
        IndexVar i_bounded("i_bounded");
        IndexVar i0("i0"), i1("i1");
        IndexExpr precomputedExpr = B(i) * C(i);
        A(i) = precomputedExpr;

        IndexStmt stmt = A.getAssignment().concretize();
        TensorVar precomputed("precomputed", Type(Float64, {Dimension(i1)}), taco::dense);
        stmt = stmt.bound(i, i_bounded, 16, BoundType::MaxExact)
                .split(i_bounded, i0, i1, 4);

        stmt = stmt.precompute(precomputedExpr, i1, i1, precomputed);

        A.compile(stmt.concretize());
        A.assemble();
        A.compute();

        Tensor<double> expected("expected", {16}, Format{Dense});
        expected(i) = B(i) * C(i);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);

        //_printToCout(stmt);

//  ir::IRPrinter irp = ir::IRPrinter(cout);
//    
//  cout << stmt << endl;
//
//  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
//  ir::Stmt compute = lower(stmt, "compute",  false, true);
//  
//  irp.print(compute);
//  cout << endl;
//  codegen->compile(compute, false);

    }

    TEST(workspaces, precompute2D_add) {
        /// GENGHAN: Target is forall(i, forall(j, A(i,j) = D(i,j))

        int N = 16;
        Tensor<double> A("A", {N, N}, Format{Dense, Dense});
        Tensor<double> B("B", {N, N}, Format{Dense, Dense});
        Tensor<double> C("C", {N, N}, Format{Dense, Dense});
        Tensor<double> D("D", {N, N}, Format{Dense, Dense});
        Tensor<double> E("E", {N, N}, Format{Dense, Dense});

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B.insert({i, j}, (double) i);
                C.insert({i, j}, (double) j);
                D.insert({i, j}, (double) i * j);
                E.insert({i,j}, (double) i * j);
            }
        }

        IndexVar i("i"), j("j");
        IndexExpr precomputedExpr = B(i, j) + C(i, j);
        A(i, j) = precomputedExpr + D(i, j) + E(i,j);
        TensorVar ws1("ws1", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});
        TensorVar ws2("ws2", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});
        TensorVar ws3("ws3", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});

        IndexStmt stmt = A.getAssignment().concretize();
        stmt = stmt.precompute(precomputedExpr, {i, j}, {i, j}, ws1);
        stmt = stmt.precompute(ws1(i,j)+D(i,j), {i, j}, {i, j}, ws2);
        stmt = stmt.precompute(ws2(i,j)+E(i,j), {i, j}, {i, j}, ws3);

        A.compile(stmt.concretize());
        A.assemble();
        A.compute();



        Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
        expected(i, j) = B(i, j) + C(i, j) + D(i, j);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);
        //_printToCout(stmt);

    }


    TEST(workspaces, precompute4D_add) {
        int N = 16;
        Tensor<double> A("A", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
        Tensor<double> B("B", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
        Tensor<double> C("C", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
        Tensor<double> D("D", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    for (int l = 0; l < N; l++) {
                        B.insert({i, j, k, l}, (double) i + j);
                        C.insert({i, j, k, l}, (double) j * k);
                        D.insert({i, j, k, l}, (double) k * l);
                    }
                }
            }
        }

        IndexVar i("i"), j("j"), k("k"), l("l");
        IndexVar iw("iw"), jw("jw"), kw("kw"), lw("lw");
        IndexExpr precomputedExpr = B(i, j, k, l) + C(i, j, k, l);
        IndexExpr pprecomputedExpr = precomputedExpr + D(i, j, k, l);
        A(i, j, k, l) = pprecomputedExpr;


        IndexStmt stmt = A.getAssignment().concretize();
        TensorVar ws1("ws1", Type(Float64, {(size_t) N, (size_t) N, (size_t) N, (size_t) N}),
                      Format{Dense, Dense, Dense, Dense});
        TensorVar ws2("ws2", Type(Float64, {(size_t) N, (size_t) N, (size_t) N, (size_t) N}),
                      Format{Dense, Dense, Dense, Dense});


        stmt = stmt.precompute(precomputedExpr, {i, j, k, l}, {i, j, k, l}, ws1)
                .precompute(ws1(i, j, k, l) + D(i, j, k, l), {i, j, k, l}, {iw, jw, kw, lw}, ws2);

        A.compile(stmt.concretize());
        A.assemble();
        A.compute();
        cout<<stmt<<endl;

        //_printToCout(stmt);
        Tensor<double> expected("expected", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
        expected(i, j, k, l) = B(i, j, k, l) + C(i, j, k, l) + D(i, j, k, l);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);
    }

    TEST(workspaces, precompute4D_multireduce) {
        int N = 16;
        Tensor<double> A("A", {N, N}, Format{Dense, Dense});
        Tensor<double> B("B", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
        Tensor<double> C("C", {N, N, N}, Format{Dense, Dense, Dense});
        Tensor<double> D("D", {N, N}, Format{Dense, Dense});

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    for (int l = 0; l < N; l++) {
                        B.insert({i, j, k, l}, (double) k * l);
                        C.insert({i, j, k}, (double) j * k);
                        D.insert({i, j}, (double) i + j);
                    }
                }
            }
        }

        IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
        IndexExpr precomputedExpr = B(i, j, k, l) * C(k, l, m);
        A(i, j) = precomputedExpr * D(m, n);


        IndexStmt stmt = A.getAssignment().concretize();
        TensorVar ws1("ws1", Type(Float64, {(size_t) N, (size_t) N, (size_t) N}), Format{Dense, Dense, Dense});
        TensorVar ws2("ws2", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});
        stmt = stmt.precompute(precomputedExpr, {i, j, m}, {i, j, m}, ws1)
                .precompute(ws1(i, j, m) * D(m, n), {i, j}, {i, j}, ws2);
        //_printToCout(stmt);
        A.compile(stmt.concretize());
        A.assemble();
        A.compute();

        Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
        expected(i, j) = B(i, j, k, l) * C(k, l, m) * D(m, n);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);

    }

    TEST(workspaces, precompute3D_TspV) {
        int N = 16;
        Tensor<double> A("A", {N, N}, Format{Dense, Dense});
        Tensor<double> B("B", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
        Tensor<double> c("c", {N}, Format{Sparse});

        for (int i = 0; i < N; i++) {
            c.insert({i}, (double) i);
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    for (int l = 0; l < N; l++) {
                        B.insert({i, j, k, l}, (double) i + j);
                    }
                }
            }
        }

        IndexVar i("i"), j("j"), k("k"), l("l");
        IndexExpr precomputedExpr = B(i, j, k, l) * c(l);
        A(i, j) = precomputedExpr * c(k);


        IndexStmt stmt = A.getAssignment().concretize();
        TensorVar ws("ws", Type(Float64, {(size_t) N, (size_t) N, (size_t) N}), Format{Dense, Dense, Dense});
        stmt = stmt.precompute(precomputedExpr, {i, j, k}, {i, j, k}, ws);
        stmt = stmt.concretize();
        //_printToCout(stmt);
        A.compile(stmt);
        A.assemble();
        A.compute();

        Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
        expected(i, j) = (B(i, j, k, l) * c(l)) * c(k);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);

    }

    TEST(workspaces, precompute3D_multipleWS) {
        int N = 16;
        Tensor<double> A("A", {N, N}, Format{Dense, Dense});
        Tensor<double> B("B", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
        Tensor<double> c("c", {N}, Format{Sparse});

        for (int i = 0; i < N; i++) {
            c.insert({i}, (double) i);
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    for (int l = 0; l < N; l++) {
                        B.insert({i, j, k, l}, (double) i + j);
                    }
                }
            }
        }

        IndexVar i("i"), j("j"), k("k"), l("l");
        IndexExpr precomputedExpr = B(i, j, k, l) * c(l);
        IndexExpr precomputedExpr2 = precomputedExpr * c(k);
        A(i, j) = precomputedExpr2;


        IndexStmt stmt = A.getAssignment().concretize();
        TensorVar ws("ws", Type(Float64, {(size_t) N, (size_t) N, (size_t) N}), Format{Dense, Dense, Dense});
        TensorVar t("t", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});
        stmt = stmt.precompute(precomputedExpr, {i, j, k}, {i, j, k}, ws);

        stmt = stmt.precompute(ws(i, j, k) * c(k), {i, j}, {i, j}, t);
        //stmt = stmt.precompute(precomputedExpr2, {i, j}, {i, j}, t);
        stmt = stmt.concretize();
        //_printToCout(stmt);
        A.compile(stmt);
        A.assemble();
        A.compute();

        Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
        expected(i, j) = (B(i, j, k, l) * c(l)) * c(k);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);

    }

    TEST(workspaces, precompute3D_renamedIVars_TspV) {
        int N = 16;
        Tensor<double> A("A", {N, N}, Format{Dense, Dense});
        Tensor<double> B("B", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
        Tensor<double> c("c", {N}, Format{Sparse});

        for (int i = 0; i < N; i++) {
            c.insert({i}, (double) i);
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    for (int l = 0; l < N; l++) {
                        B.insert({i, j, k, l}, (double) i + j);
                    }
                }
            }
        }

        IndexVar i("i"), j("j"), k("k"), l("l");
        IndexExpr precomputedExpr = B(i, j, k, l) * c(l);
        A(i, j) = precomputedExpr * c(k);


        IndexStmt stmt = A.getAssignment().concretize();
        TensorVar ws("ws", Type(Float64, {(size_t) N, (size_t) N, (size_t) N}),
                     Format{Dense, Dense, Dense});

        IndexVar iw("iw"), jw("jw"), kw("kw");
        stmt = stmt.precompute(precomputedExpr, {i, j, k}, {iw, jw, kw}, ws);
        stmt = stmt.concretize();


        A.compile(stmt);
        A.assemble();
        A.compute();

        Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
        expected(i, j) = (B(i, j, k, l) * c(l)) * c(k);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);

    }

    TEST(workspaces, tile_dotProduct_1) {
        // FIXME: Disabled because currently the precompute algorithm does not appropriately
        //        find the correct forall substmt to next the WhereNode in after i has been
        //        split into i0 and i1. As an example, the first precompute below is incorrect
        //        since it should transform
        //        forall(i0, forall(i1, A() += B(i) * C(i))) -->
        //        forall(i0, where(forall(i1, A() = ws(i1)), forall(i1, ws(i1) += B(i) * C(i))))
        //
        //        But currently the algorithm does
        //        forall(i0, forall(i1, A() += B(i) * C(i))) -->
        //        where(forall(i1, A() += ws(i1)), forall(i0, forall(i1, ws(i1) += B(i) * C(i))))

        int N = 1024;
        Tensor<double> A("A");
        Tensor<double> B("B", {N}, Format({Dense}));
        Tensor<double> C("C", {N}, Format({Dense}));

        for (int i = 0; i < N; i++) {
            B.insert({i}, (double) i);
            C.insert({i}, (double) i);
        }

        B.pack();
        C.pack();

        IndexVar i("i");
        IndexVar i_bounded("i_bounded");
        IndexVar i0("i0"), i1("i1");
        IndexExpr BExpr = B(i);
        IndexExpr CExpr = C(i);
        IndexExpr precomputedExpr = (BExpr) * (CExpr);
        A() = precomputedExpr;

        IndexStmt stmt = A.getAssignment().concretize();
        TensorVar B_new("B_new", Type(Float64, {(size_t) N}), taco::dense);
        TensorVar C_new("C_new", Type(Float64, {(size_t) N}), taco::dense);
        TensorVar precomputed("ws", Type(Float64, {(size_t) N}), taco::dense);

        stmt = stmt.bound(i, i_bounded, (size_t) N, BoundType::MaxExact)
                .split(i_bounded, i0, i1, 32)
                .reorder({i0,i1});
        stmt = stmt.precompute(precomputedExpr, i1, i1, precomputed);
        stmt = stmt.precompute(BExpr, i1, i1, B_new)
                .precompute(CExpr, i1, i1, C_new);

        stmt = stmt.concretize();

        A.compile(stmt);
        A.assemble();
        A.compute();

        ir::IRPrinter irp = ir::IRPrinter(cout);

        cout << stmt << endl;

        std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
        ir::Stmt compute = lower(stmt, "compute", false, true);

        irp.print(compute);
        cout << endl;
        codegen->compile(compute, false);

        Tensor<double> expected("expected");
        expected() = B(i) * C(i);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);
    }

    TEST(workspaces, tile_dotProduct_2) {
        // FIXME: This is also currently disabled since split(...) scheduling commands
        // only split on the FIRST INSTANCE of an indexVar (assumes only one).
        // This is wrong if the indexVar is not renamed across iw_vars since an indexVar can
        // then occur on BOTH the consumer and producer side and should be split across both.

        int N = 1024;
        Tensor<double> A("A");
        Tensor<double> B("B", {N}, Format({Dense}));
        Tensor<double> C("C", {N}, Format({Dense}));

        for (int i = 0; i < N; i++) {
            B.insert({i}, (double) i);
            C.insert({i}, (double) i);
        }

        B.pack();
        C.pack();

        IndexVar i("i");
        IndexVar i_bounded("i_bounded");
        IndexVar i0("i0"), i1("i1");
        IndexExpr BExpr = B(i);
        IndexExpr CExpr = C(i);
        IndexExpr precomputedExpr = (BExpr) * (CExpr);
        A() = precomputedExpr;

        IndexStmt stmt = A.getAssignment().concretize();
        TensorVar B_new("B_new", Type(Float64, {(size_t) N}), taco::dense);
        TensorVar C_new("C_new", Type(Float64, {(size_t) N}), taco::dense);
        TensorVar precomputed("precomputed", Type(Float64, {(size_t) N}), taco::dense);

        stmt = stmt.precompute(precomputedExpr, i, i, precomputed);

        stmt = stmt.precompute(BExpr, i, i, B_new)
                .precompute(CExpr, i, i, C_new);

        stmt = stmt.bound(i, i_bounded, (size_t) N, BoundType::MaxExact)
                .split(i_bounded, i0, i1, 32);

        stmt = stmt.concretize();

        A.compile(stmt);
        A.assemble();
        A.compute();

        Tensor<double> expected("expected");
        expected() = B(i) * C(i);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);
    }

    TEST(workspaces, tile_dotProduct_3) {
        int N = 1024;
        Tensor<double> A("A");
        Tensor<double> B("B", {N}, Format({Dense}));
        Tensor<double> C("C", {N}, Format({Dense}));

        for (int i = 0; i < N; i++) {
            B.insert({i}, (double) i);
            C.insert({i}, (double) i);
        }

        B.pack();
        C.pack();

        IndexVar i("i");
        IndexVar i_bounded("i_bounded");
        IndexVar i0("i0"), i1("i1");
        IndexExpr BExpr = B(i);
        IndexExpr CExpr = C(i);
        IndexExpr precomputedExpr = (BExpr) * (CExpr);
        A() = precomputedExpr;

        IndexStmt stmt = A.getAssignment().concretize();
        TensorVar B_new("B_new", Type(Float64, {(size_t) N/32}), taco::dense);
        TensorVar C_new("C_new", Type(Float64, {(size_t) N/32}), taco::dense);
        TensorVar precomputed("precomputed", Type(Float64, {(size_t) N/32}), taco::dense);

        stmt = stmt.bound(i, i_bounded, (size_t) N, BoundType::MaxExact)
                .split(i_bounded, i0, i1, 32);
        stmt = stmt.precompute(precomputedExpr, i0, i0, precomputed);


        stmt = stmt.precompute(BExpr, i1, i1, B_new)
                .precompute(CExpr, i1, i1, C_new);

        //_printToCout(stmt);
        stmt = stmt.concretize();
        cout<<stmt<<endl;

        //_printToFile("tile_dP_3",stmt);
        A.compile(stmt);
        A.assemble();
        A.compute();

        Tensor<double> expected("expected");
        expected() = B(i) * C(i);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);


    }

    TEST(workspaces, DISABLED_facilitate_reorder) {
        int N = 256;
        int M = 256;
        Tensor<float> A("A",{M},Format({Dense}));
        Tensor<float> B("B", {M,N}, CSC);
        Tensor<float> C("C", {N}, Format({Dense}));
        Tensor<float> D("D", {M}, Format({Dense}));
        float SPARSITY = .3;
        for (int i = 0; i < N; i++) {
            for (int j=0;j<M;j++) {
                float rand_float = (float) rand() / (float) (RAND_MAX);
                B.insert({j, i}, (float) ((int) (rand_float * 3 / SPARSITY)));
            }
        }
        for (int i=0;i<N;i++){
            C.insert({i},(float ) i);
        }
        for (int i=0;i<M;i++){
            D.insert({i},(float ) i);
        }
        IndexVar i("i"),j("j");
        IndexExpr precomputedExpr=B(i,j) * C(j);
        A(i) = precomputedExpr + D(i);
        TensorVar precomputed("precomputed", Type(Float64, {(size_t) M}), taco::dense);
        IndexStmt stmt = A.getAssignment().concretize();
        stmt.precompute(precomputedExpr,i,i,precomputed);
        stmt = stmt.concretize();
        A.compile(stmt);
        A.assemble();
        A.compute();

        Tensor<float> expected("expected");
        expected(i) = B(i,j) * C(j)+D(i);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);


    }

    TEST(workspaces, test_ori_same){
        //GENGHAN:i1temp has the same name as i1
        //        forall(i0, forall(i1, A() += B(i) * C(i))) -->
        //        where(forall(i1, A() += ws(i1)), forall(i0, forall(i1, ws(i1) += B(i) * C(i))))

        int N = 1024;
        Tensor<double> A("A");
        Tensor<double> B("B", {N}, Format({Dense}));
        Tensor<double> C("C", {N}, Format({Dense}));

        for (int i = 0; i < N; i++) {
            B.insert({i}, (double) i);
            C.insert({i}, (double) i);
        }

        B.pack();
        C.pack();

        IndexVar i("i");
        IndexVar i_bounded("i_bounded");
        IndexVar i0("i0"), i1("i1"),i1tmp("i1");
        IndexExpr BExpr = B(i);
        IndexExpr CExpr = C(i);
        IndexExpr precomputedExpr = (BExpr) * (CExpr);
        A() = precomputedExpr;

        IndexStmt stmt = A.getAssignment().concretize();
        TensorVar precomputed("ws", Type(Float64, {(size_t) 32}), taco::dense);

        stmt = stmt.bound(i, i_bounded, (size_t) N, BoundType::MaxExact)
                .split(i_bounded, i0, i1, 32)
                .reorder({i0,i1});
        stmt = stmt.precompute(precomputedExpr, i1, i1tmp, precomputed);

        stmt = stmt.concretize();

        A.compile(stmt);
        A.assemble();
        A.compute();

        ir::IRPrinter irp = ir::IRPrinter(cout);

        cout << stmt << endl;

        std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
        ir::Stmt compute = lower(stmt, "compute", false, true);

        irp.print(compute);
        cout << endl;
        codegen->compile(compute, false);

        Tensor<double> expected("expected");
        expected() = B(i) * C(i);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);
    }
    TEST(workspaces, test_ori_diff){
        //GENGHAN:i1temp has different name with i1
        //        forall(i0, forall(i1, A() += B(i) * C(i))) -->
        //        where(forall(i1, A() += ws(i1)), forall(i0, forall(i1, ws(i1) += B(i) * C(i))))

        int N = 1024;
        Tensor<double> A("A");
        Tensor<double> B("B", {N}, Format({Dense}));
        Tensor<double> C("C", {N}, Format({Dense}));

        for (int i = 0; i < N; i++) {
            B.insert({i}, (double) i);
            C.insert({i}, (double) i);
        }

        B.pack();
        C.pack();

        IndexVar i("i");
        IndexVar i0("i0"), i1("i1"),i1tmp("i1tmp");
        IndexExpr BExpr = B(i);
        IndexExpr CExpr = C(i);
        IndexExpr precomputedExpr = (BExpr) * (CExpr);
        A() = precomputedExpr;

        IndexStmt stmt = A.getAssignment().concretize();
        TensorVar precomputed("ws", Type(Float64, {(size_t) 32}), taco::dense);

        stmt = stmt//.bound(i, i_bounded, (size_t) N, BoundType::MaxExact)
                .split(i, i0, i1, 32)
                .reorder({i0,i1});
        stmt = stmt.precompute(precomputedExpr, i1, i1tmp, precomputed);

        stmt = stmt.concretize();

        A.compile(stmt);
        A.assemble();
        A.compute();

        ir::IRPrinter irp = ir::IRPrinter(cout);

        cout << stmt << endl;

        std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
        ir::Stmt compute = lower(stmt, "compute", false, true);

        irp.print(compute);
        cout << endl;
        codegen->compile(compute, false);

        Tensor<double> expected("expected");
        expected() = B(i) * C(i);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);
    }
    TEST(workspaces, test_new_same){
        //GENGHAN:i1temp has the same name as i1
        //        forall(i0, forall(i1, A() += B(i) * C(i))) -->
        //        forall(i0, where(forall(i1, A() += ws(i1)), forall(i1, ws(i1) += B(i) * C(i))))
        //        Use the hack part in transformations.cpp

        int N = 1024;
        Tensor<double> A("A");
        Tensor<double> B("B", {N}, Format({Dense}));
        Tensor<double> C("C", {N}, Format({Dense}));

        for (int i = 0; i < N; i++) {
            B.insert({i}, (double) i);
            C.insert({i}, (double) i);
        }

        B.pack();
        C.pack();

        IndexVar i("i"),i0("i0"), i1("i1"),i1tmp("i1");
        IndexExpr BExpr = B(i);
        IndexExpr CExpr = C(i);
        IndexExpr precomputedExpr = (BExpr) * (CExpr);
        A() = precomputedExpr;

        IndexStmt stmt = A.getAssignment().concretize();
        TensorVar precomputed("ws", Type(Float64, {(size_t) 32}), taco::dense);

        stmt = stmt
                .split(i, i0, i1, 32)
                .reorder({i0,i1})
                .precompute(precomputedExpr, i1, i1tmp, precomputed);

        stmt = stmt.concretize();

        A.compile(stmt);
        A.assemble();
        A.compute();

        //_printToFile("success_tmp",stmt);

        Tensor<double> expected("expected");
        expected() = B(i) * C(i);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);
    }
    TEST(workspaces, DISABLED_test_new_diff){
        //GENGHAN:i1temp has different name with i1
        //        forall(i0, forall(i1, A() += B(i) * C(i))) -->
        //        forall(i0, where(forall(i1, A() += ws(i1)), forall(i1, ws(i1) += B(i) * C(i))))
        //        Use the hack part in transformations.cpp

        int N = 1024;
        Tensor<double> A("A");
        Tensor<double> B("B", {N}, Format({Dense}));
        Tensor<double> C("C", {N}, Format({Dense}));

        for (int i = 0; i < N; i++) {
            B.insert({i}, (double) i);
            C.insert({i}, (double) i);
        }

        B.pack();
        C.pack();

        IndexVar i("i"),i0("i0"), i1("i1"),i1tmp("i1tmp");
        IndexExpr precomputedExpr = B(i) * C(i);
        A() = precomputedExpr;

        IndexStmt stmt = A.getAssignment().concretize();
        TensorVar precomputed("ws", Type(Float64, {(size_t) 32}), taco::dense);

        stmt = stmt
                .split(i, i0, i1, 32)
                .reorder({i0,i1})
                .precompute(precomputedExpr, i1, i1tmp, precomputed);

        stmt = stmt.concretize();

        A.compile(stmt);
        A.assemble();
        A.compute();

        //_printToFile("fail_tmp",stmt);

        Tensor<double> expected("expected");
        expected() = B(i) * C(i);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);
    }

    TEST(workspaces, DISABLED_chain_rule_fail_1) {
        /// From workspace paper: If S2 modifies its tensor with an assignment statement, then (Any S1) where (Any S2) is equivalent with Any(S1 where S2)
        /// FIXME: use of undeclared identifier 'jws1'

        int N = 16;
        Tensor<double> A("A", {N, N}, Format{Dense, Dense});
        Tensor<double> B("B", {N, N}, Format{Dense, Dense});
        Tensor<double> C("C", {N, N}, Format{Dense, Dense});
        Tensor<double> D("D", {N, N}, Format{Dense, Dense});

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B.insert({i, j}, (double) i);
                C.insert({i, j}, (double) j);
                D.insert({i, j}, (double) i * j);
            }
        }

        IndexVar i("i"), j("j"),io("io"),jo("jo");
        IndexExpr precomputedExpr = B(i, j) + C(i, j);
        A(i, j) = precomputedExpr + D(i, j);
        TensorVar ws1("ws1", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});
        TensorVar ws2("ws2", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});

        TensorVar Av("A", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});
        TensorVar Bv("B", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});
        TensorVar Cv("C", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});
        TensorVar Dv("D", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});


        IndexStmt stmt= where(forall(i,
                                      forall(j,
                                             Av(i,j)=ws2(i,j))),
                               forall(i,
                                      forall(j,
                                             where(
                                                     ws2(i,j) = ws1(i,j) + Dv(i,j),
                                                     ws1(i,j) = Bv(i,j) + Cv(i,j)))));

        /*
        IndexStmt stmt= where(forall(i,
                                     forall(j,
                                            Av(i,j)=ws2(i,j))),
                              where(forall(i, forall(j, ws2(i,j) = ws1(i,j) + Dv(i,j))), forall(i,
                                                                                                forall(j, ws1(i,j) = Bv(i,j) + Cv(i,j)))

                              ));
        */
        cout<<stmt<<endl;
        //_printToFile("fail_chain",stmt);
        A.compile(stmt.concretize());
        A.assemble();
        A.compute();

        Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
        expected(i, j) = B(i, j) + C(i, j) + D(i, j);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);

    }

    TEST(workspaces, chain_rule_fail_2) {
        /// FIXME: The expression (ws1(i,j) + D(i,j)) is not in forall(i, forall(j, A(i,j) = B(i,j) + C(i,j) + D(i,j)))

        int N = 16;
        Tensor<double> A("A", {N, N}, Format{Dense, Dense});
        Tensor<double> B("B", {N, N}, Format{Dense, Dense});
        Tensor<double> C("C", {N, N}, Format{Dense, Dense});
        Tensor<double> D("D", {N, N}, Format{Dense, Dense});

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B.insert({i, j}, (double) i);
                C.insert({i, j}, (double) j);
                D.insert({i, j}, (double) i * j);
            }
        }

        IndexVar i("i"), j("j");
        IndexExpr precomputedExpr = B(i, j) + C(i, j);
        A(i, j) = precomputedExpr + D(i, j);
        TensorVar ws1("ws1", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});
        TensorVar ws2("ws2", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});

        TensorVar Av("A", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});
        TensorVar Bv("B", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});
        TensorVar Cv("C", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});
        TensorVar Dv("D", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});


        IndexStmt stmt = forall(i,
                                 forall(j,
                                        Av(i,j) = (Bv(i,j)+Cv(i,j))+Dv(i,j)));
        stmt = stmt.precompute(Bv(i,j)+Cv(i,j), {i, j}, {i, j}, ws1);
        stmt = stmt.precompute(ws1(i,j)+Dv(i,j), {i, j}, {i, j}, ws2);

        A.compile(stmt.concretize());
        A.assemble();
        A.compute();

        Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
        expected(i, j) = B(i, j) + C(i, j) + D(i, j);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);
        //_printToCout(stmt);
    }

    TEST(workspaces, chain_rule) {
        /// GENGHAN: Exactly How to generate correct foralls before consumer?

        int N = 16;
        Tensor<double> A("A", {N, N}, Format{Dense, Dense});
        Tensor<double> B("B", {N, N}, Format{Dense, Dense});
        Tensor<double> C("C", {N, N}, Format{Dense, Dense});
        Tensor<double> D("D", {N, N}, Format{Dense, Dense});

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B.insert({i, j}, (double) i);
                C.insert({i, j}, (double) j);
                D.insert({i, j}, (double) i * j);
            }
        }

        IndexVar i("i"), j("j");
        IndexExpr precomputedExpr = B(i, j) + C(i, j);
        A(i, j) = precomputedExpr + D(i, j);
        TensorVar ws1("ws1", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});
        TensorVar ws2("ws2", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});

        TensorVar Av("A", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});
        TensorVar Bv("B", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});
        TensorVar Cv("C", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});
        TensorVar Dv("D", Type(Float64, {(size_t) N, (size_t) N}), Format{Dense, Dense});


        IndexStmt stmt= where(forall(i,
                                      forall(j,
                                             Av(i,j)=ws2(i,j))),
                               where(
                                       forall(i,
                                              forall(j,
                                                     ws2(i,j) = ws1(i,j) + Dv(i,j))),
                                       forall(i,
                                              forall(j,
                                                     ws1(i,j) = Bv(i,j) + Cv(i,j)))));

        A.compile(stmt.concretize());
        A.assemble();
        A.compute();

        Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
        expected(i, j) = B(i, j) + C(i, j) + D(i, j);
        expected.compile();
        expected.assemble();
        expected.compute();
        ASSERT_TENSOR_EQ(expected, A);
        //_printToCout(stmt);
    }




}


