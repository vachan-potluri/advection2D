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
 * 1. Mesh is setup and stored in triang
 * 2. dof_handler is linked to fe
 * 3. Global solution and local rhs sizes are set
 * 4. Cell user indices are set (for owner/neighbor distinction)
 * 5. Sizes of stiffness and lifting matrix containers are set
 */
void advection2D::setup_system()
{
        // initialise the triang variable
        GridGenerator::hyper_cube(triang);
        triang.refine_global(5); // 2^5=32 cells in each direction

        // set dof_handler
        dof_handler.distribute_dofs(fe);

        // no system_matrix because the solution is updated cell wise
        g_solution.reinit(dof_handler.n_dofs());
        gold_solution.reinit(dof_handler.n_dofs());
        l_rhs.reinit(fe.dofs_per_cell);

        // set user flags for cell
        // for a face, cell with lower user index will be treated owner
        uint i=0;
        for(auto &cell: dof_handler.active_cell_iterators()){
                cell->set_user_index(i++);
        } // loop over cells

        // set sizes of stiffness and lifting matrix containers
        stiff_mats.reserve(triang.n_active_cells());
        lift_mats.reserve(triang.n_active_cells());
}

/**
 * @brief Assembles the system
 */
void advection2D::assemble_system()
{
        // allocate all local matrices
        FullMatrix<double> l_mass(fe.dofs_per_cell),
                l_mass_inv(fe.dofs_per_cell),
                l_diff(fe.dofs_per_cell),
                l_flux(fe.dofs_per_cell); // initialise with square matrix size
        QGauss<2> cell_quad_formula(fe.degree+1); // (N+1) gauss quad for cell
        QGauss<1> face_quad_formula(fe.degree+1); // for face
        FEValues<2> fe_values(fe, cell_quad_formula,
                update_values | update_gradients | update_JxW_values);
        FEFaceValues<2> fe_face_values(fe, face_quad_formula,
                update_values | update_JxW_values);
        
        // compute mass and diff matrices
        for(auto &cell: dof_handler.active_cell_iterators()){
                fe_values.reinit(cell);
                l_mass = 0;
                l_diff = 0;
                for(uint qid=0; qid<fe_values.n_quadrature_points; qid++){
                        for(uint i=0; i<fe_values.dofs_per_cell; i++){
                                for(uint j=0; j<fe_values.dofs_per_cell; j++){
                                        l_mass(i,j) += fe_values.shape_value(i, qid) *
                                                fe_values.shape_value(j, qid) *
                                                fe_values.JxW(qid);
                                        l_diff(i,j) += fe_values.shape_grad(i, qid) *
                                                wind(fe_values.quadrature_point(qid)) *
                                                fe_values.shape_value(j, qid) *
                                                fe_values.JxW(qid);
                                }
                        }
                } // loop over quad points
        }// loop over cells
}
