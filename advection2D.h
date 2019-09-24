/**
 * @file advection2D.h
 * @brief Defines advection2D class
 */

// Includes: most of them are from step-12 and dflo
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition_block.h>

// #include <deal.II/numerics/derivative_approximation.h> // for adaptive mesh

#include "common.h"
#include "wind.h"

#ifndef advection2D_h
#define advection2D_h

/**
 * @class advection2D
 * @brief This is mostly a blend of step-12 and dflo code
 * 
 * The problem to be solved is
 * @f[ \frac{\partial \phi}{\partial t} + \nabla \cdot (\phi \vec{v}) = 0 @f]
 * @f$ \phi @f$ is the variable and @f$\vec{v}@f$ is the advecting wind. The DG formulation of this
 * problem is
 * @f[
 * \int_{\Omega_h} l_j \left(\sum \frac{\partial\phi_i}{\partial t} l_i\right) \,d\Omega +
 * \int_{\partial\Omega_h}l_j \left(\sum\phi^*_i l_i\right) \vec{v}\cdot\vec{n}\,dA -
 * \int_{\Omega_h}\nabla l_j\cdot\vec{v} \left(\sum\phi_i l_i\right) \,d\Omega = 0
 * @f]
 * Explicit time integration gives
 * @f[
 * [M]\{\phi\}^{n+1} = [M]\{\phi\}^n + \left( [D]\{\phi\}^n - [F]\{f^*\}^n \right)\Delta t
 * @f]
 * @f$[M]@f$ is the mass matrix, @f$[D]@f$ is the differentiation matrix and @f$[F]@f$ is the flux
 * matrix. Multiplying by mass inverse:
 * @f[
 * \{\phi\}^{n+1} = \{\phi\}^n + \left( [S]\{\phi\}^n - [L]\{f^*\}^n \right)
 * @f]
 * Here @f$[S]@f$ is the stiffness matrix and @f$[L]@f$ is the lifting matrix
 */

class advection2D
{

        public:
        advection2D(const int order);

        private:
        void setup_system();
        void assemble_system();

        // class variables
        Triangulation<2> triang;
        const MappingQ1<2> mapping;

        FE_DGQ<2> fe;
        DoFHandler<2> dof_handler;

        // solution has to be global to enable results output, a local solution cannot used to
        // output results
        Vector<double> g_solution; // global solution
        Vector<double> gold_solution; // global old solution
        Vector<double> l_rhs; // local rhs (to be updated for every cell)

        // stiffness and lifting matrices
        std::vector<FullMatrix<double>> stiff_mats;
        std::vector<FullMatrix<double>> lift_mats;



        public:
        #ifdef DEBUG
        static void test();
        #endif
};

#endif
