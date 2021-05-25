#ifndef TACO_CODEGEN_LEGION_H
#define TACO_CODEGEN_LEGION_H

#include "codegen.h"
#include "taco/ir/ir.h"

namespace taco {
namespace ir {

class CodegenLegion : virtual public CodeGen {
public:
  virtual ~CodegenLegion() = default;
  CodegenLegion() {}

  std::string unpackTensorProperty(std::string varname, const GetProperty* op, bool is_output_prop) override;
  std::string printFuncName(const Function *func,
                            std::map<Expr, std::string, ExprCompare> inputMap,
                            std::map<Expr, std::string, ExprCompare> outputMap) override;

  std::string taskArgsName(std::string taskName) {
    return taskName + "Args";
  }

  struct AccessorInfo {
    TensorProperty prop;
    int dims;
    Datatype typ;

    friend bool operator<(const AccessorInfo& a, const AccessorInfo& b) {
      if (a.prop < b.prop) {
        return true;
      }
      if (a.dims < b.dims) {
        return true;
      }
      if (a.typ < b.typ) {
        return true;
      }
      return false;
    }
  };
  virtual void emitHeaders(std::ostream& out);
  void collectAndEmitAccessors(ir::Stmt stmt, std::ostream& out);
  void collectAllFunctions(ir::Stmt stmt);

  // This must be called after collectAllFunctions.
  void rewriteFunctionTaskIDs();

  // This must be called after rewriteFunctionTaskIDs. This function
  // will do a large amount of work of transforming task for loops
  // into actual tasks, and finding out the variables used by each
  // task.
  void analyzeAndCreateTasks(std::ostream& out);
  void emitRegisterTasks(std::ostream& out);
  virtual std::string procForTask(Stmt target, Stmt func);

  static std::string getVarName(Expr e) {
    if (isa<Var>(e)) {
      return e.as<Var>()->name;
    }
    if (isa<GetProperty>(e)) {
      return e.as<GetProperty>()->name;
    }
    taco_ierror;
    return "";
  }

  static Datatype getVarType(Expr e) {
    if (isa<Var>(e)) {
      return e.as<Var>()->type;
    }
    if (isa<GetProperty>(e)) {
      return e.as<GetProperty>()->type;
    }
    taco_ierror;
    return Datatype();
  }

  std::string accessorType(const GetProperty* op) {
    switch (op->property) {
      case TensorProperty::ValuesReadAccessor: {
        std::stringstream ss;
        ss << "AccessorRO" << printType(op->type, false) << op->mode;
        return ss.str();
      }
      case TensorProperty::ValuesWriteAccessor: {
        std::stringstream ss;
        ss << "AccessorRW" << printType(op->type, false) << op->mode;
        return ss.str();
      }
      case TensorProperty::ValuesReductionAccessor: {
        std::stringstream ss;
        ss << "AccessorReduce" << printType(op->type, false) << op->mode;
        return ss.str();
      }
      default:
        taco_iassert(false);
        return "";
    }
  }

  // Fields for code generation.
  std::set<AccessorInfo> accessors;
  std::vector<Stmt> allFunctions;

  // Map of created function to region arguments.
  std::map<Stmt, std::vector<Expr>> regionArgs;
  // Map from each target codegen function to the child task functions that
  // are generated by each task.
  std::map<Stmt, std::vector<Stmt>> functions;

  // Map from TaskIDs to For* and Function* corresponding to those TaskID's.
  std::map<int, Stmt> idToFor;
  std::map<int, Stmt> idToFunc;

  // Map to from Function* to the For* that is it's body.
  std::map<Stmt, Stmt> funcToFor;
  // Map from Function* to the codegen target function that it is generated for.
  std::map<Stmt, Stmt> funcToParentFunc;
  // Map from tasks to the region arguments the task uses.
  std::map<Stmt, std::vector<Expr>> taskArgs;
};

}
}

#endif //TACO_CODEGEN_LEGION_H
