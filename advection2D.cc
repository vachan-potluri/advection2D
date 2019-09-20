/**
 * @file advection2D.cc
 * @brief Defines advection2D class
 */

#include "advection2D.h"

/**
 * @brief Constructor with order of polynomial approx as arg
 */
advection2D::advection2D(const int order)
: mapping(), fe(order), dof_handler(triang) // @suppress("Symbol is not resolved")
{}

/**
 * @brief Sets up the system
 * 
 * 1. Initialises advection2D::triang
 * 2. Distributes dofs of advection2D::dof_handler using advection2D::fe
 * 3. Sets size of advection2D::rhs, size and sparsity pattern of advection2D::system_matrix
 */
void advection2D::setup_system()
{
        // initialise the triang variable
        GridGenerator::hyper_cube(triang);
        triang.refine_global(5); // 2^5=32 cells in each direction

        dof_handler.distribute_dofs(fe);

        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit(sparsity_pattern);
        solution.reinit(dof_handler.n_dofs());
        rhs.reinit(dof_handler.n_dofs());
}

/**
 * @brief Assembles the system using MeshWorker
 */
void advection2D::assemble_system()
{
        MeshWorker::IntegrationInfoBox<2> info_box;
        
        const uint n_quad_pts = fe.degree + 1;
        // set quad points for cells, interior and boundary faces
        info_box.initialize_gauss_quadrature(n_quad_pts, n_quad_pts, n_quad_pts);
        info_box.initialize_update_flags();
        UpdateFlags update_flags = update_quadrature_points | update_values | update_gradients;
        info_box.add_update_flags(update_flags, true, true, true, true);

        info_box.initialize(fe, mapping);

        dofInfo dof_info(dof_handler);

        MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double>> assembler;
        assembler.initialize(system_matrix, rhs);
        

        // loop
}

/**
 * @brief Integrates cell term
 * 
 * Performs the integral
 * @f[
 * \int_{\Omega_h} \left( l_i l_j + \nabla l_i \cdot \vec{v} l_j \right)\,d\Omega
 * @f]
 * for a given cell. The above expression, multiplied by the vector of dof values gives the current
 * cell term.
 * @note This function is static because it doesn't depend on class instance variables
 */
void advection2D::integrate_cell_term(dofInfo &dinfo, cellInfo &cinfo)
{
        const FEValuesBase<2> & fe_values = cinfo.fe_values();
        FullMatrix<double> & local_matrix = dinfo.matrix(0).matrix;
        const vector<double> &JxW = fe_values.get_JxW_values();

        // loop over quad points. pid is the point id
        for(uint pid=0; pid<fe_values.n_quadrature_points; pid++){
                const Tensor<1,2> current_wind = wind(fe_values.quadrature_point(pid));
                for(uint i=0; i<fe_values.dofs_per_cell; i++){
                        for(uint j=0; j<fe_values.dofs_per_cell; j++){
                                local_matrix(i,j) += ( fe_values.shape_value(i, pid) *
                                        fe_values.shape_value(j, pid) + // mass contrib
                                        current_wind * fe_values.shape_grad(i, pid) *
                                        fe_values.shape_value(j, pid) ) * JxW[pid]; // differentiation contrib
                        }
                }
        }
}
