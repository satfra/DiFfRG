// The old header and the old type name must both keep working, and using the old name must
// tell the user what to write instead.
#include <DiFfRG/common/json.hh>

double probe(const DiFfRG::JSONValue &json) { return json.get_double("/physical/Lambda"); }
