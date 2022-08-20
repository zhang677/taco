#include "taco/lower/mode_format_coord.h"

using namespace std;
using namespace taco::ir;

namespace taco {

CoordModeFormat::CoordModeFormat() :
  ModeFormatImpl("wspace", true, false, false, false, true, true,
             true, false, false, true, true, false, false, false, true) {
}
ModeFormat CoordModeFormat::copy(
  std::vector<ModeFormat::Property> properties) const {
  return ModeFormat(std::make_shared<CoordModeFormat>());
}
vector<Expr> CoordModeFormat::getArrays(Expr tensor, int mode,
                                        int level) const {
  return vector<Expr>();
}
Stmt CoordModeFormat::getTryInsert(ir::Expr insertFail, ir::Expr accumulator, ir::Expr accumulator_size
  , ir::Expr accumulator_capacity, ir::Expr crds, ir::Expr expression, Datatype type) const {
  return ir::Assign::make(accumulator_size,ir::Call::make("TryInsert_coord",{insertFail, accumulator, accumulator_size
  ,accumulator_capacity, crds, expression},type));
}
}

