#include <DiFfRG/physics/integration/abstract_integrator.hh>

namespace DiFfRG
{
  size_t AbstractIntegrator::next_integrator_id()
  {
    static size_t counter = 0;
    return counter++;
  }
} // namespace DiFfRG
