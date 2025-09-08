from ..lib import *

class minimizers():
    def __init__():
        pass

    def fun_of_repX(self, X, fun, A):
        """ 
        Get the full-likelihood by repeating the node parameters for the streq nodes. 
        Note: 
        1) these nodes are without the det_nodes, i.e. fc and fd nodes 
        2) the X, after repetition, will have ndim > 1 if dimBCX > 1
        """
        repeated_fun = self.wrappper_streq_copy(fun)
        return repeated_fun(X, A = A, verbose = False)

    def grad_of_repX(self, X, gradf, A):
        """ 
        Get the full-gradient of the likelihood by repeating the node parameters for the streq nodes. 
        Note: these nodes are without the det_nodes, i.e. fc and fd nodes 
        """
        fun4repX = self.fun_of_repX(X, fun = gradf, A = A)
        vect_fun4repX = fun4repX.reshape(-1, self.dimBCX)
        return vect_fun4repX[self.strineq_nodes].ravel()

    def _set_minimizers(self, A, verbose = False, lr = 1e-2, n_particles = 1, track_nll_opt = True):
        """
        Set the nll and grad_nll functions that will be usefull for every model
        Note: in the LPCA, the authors refer to B, C as B = X.reshape(-1, dimBCX, order = 'F')[:, dimB] (row-first) and the remaining for C.
        Here, to accomodate for both LPCA and maxlMSM, we interpret B = X.reshape(-1, dimBCX, order = 'C')[:, :dimB] (col-first) and the remaining for C.
        Practically, the .ravel and .reshape will have the same (hidden) order = 'C' parameter.
        """
        self._set_initial_guess(obs_mat = A)

        # Initialzie what we've found so far
        self.X = self.X0
        if self.X0.ndim > 1: print(f"Warning: self.X0.ndim = {self.X0.ndim}", )

        # change the loss functions if we need to consider the reduction based on inequivalent classes
        self.n_notdet_nodes = self.n_nodes - self.n_fcfd_nodes
        if self.get("reduced_by") == "neigh":
            self.nll_X = lambda X: self.fun_of_repX(X = X.ravel(), fun = self.nll, A = A)
            self.grad_nll_X = lambda X: self.grad_of_repX(X.ravel(), gradf = self.grad_nll, A = A)
        elif self.name.endswith("CM"):
            self.nll_X = lambda X: self.nll(X.ravel(), A = A, verbose = False)
            self.grad_nll_X = lambda X: self.grad_nll(X.ravel(), A = A, verbose = False).ravel()
        else:
            self.nll_X = lambda X: self.nll(X.ravel(), A = A, verbose = False)
            self.grad_nll_X = lambda X: self.grad_nll(X.ravel(), A = A, verbose = False).ravel()

        # fix learning rate for Adam
        self.learning_rate = lr

        # Keep track of the minimization process
        self.nit_list = [0]
        self.track_nll_opt = track_nll_opt
        if track_nll_opt:

            # update the list with the parameters to be optimized
            self.dXn = [self.X0]

            # update the loss and gradn
            self.losses_list, self.gradn_list,  = [self.nll_X(self.X0)], [np.linalg.norm(self.grad_nll_X(self.X0))]
            self.opt_names, self.opt_final_loss = [], []

    def minimize(self, A, n_epochs = None, opt_method = None, loss_iters = 200, verbose = False, ftol = 1e-8):
        ratio_update_lr = 3
        print(f'\n-Optimizing {self.name} with {opt_method}',)

        self.exit_opt = False
        
        for i in range(n_epochs):
            # here I set the initial guess to the current X 
            # In the next iteration put that X as initial condition 
            prev_X = self.X0.copy()

            # copy the prev_ vars to stop if we will not diminishing the likelihood
            prev_loss = self.nll_X(prev_X)
            prev_gradn = np.linalg.norm(self.grad_nll_X(prev_X))


            print(f"\n-Optimization epoch {i} / {n_epochs}", )
            print(f'-Starting loss {prev_loss:.8e}, gradn: {prev_gradn:.8e}',)
            
            if opt_method == "adam":
                res = self.adam(
                        fun = self.nll_X, 
                        X0 = self.X0, #self.mat_to_flat(mat_X = self.X0), 
                        jac = self.grad_nll_X, 
                        verbose = verbose,
                        ftol = ftol, iters = 1e2, ratio_update_lr = 3,
                        )
                
                if self.exit_opt:
                    #self.X, self.grad = res.X, res.jac
                    # self.grad = self.flat_to_mat(res.jac[:self.n_params_X])
                    break

            else:
                res = self.scipy_nll_min(opt_method = opt_method, loss_iters=loss_iters, verbose = verbose)
                
                if not self.nit:
                    curr_gradn = np.linalg.norm(self.grad)
                    break
            
            # save the results especially self.X0 will be the initial guess for the next epoch
            self.X0 = res.x.copy() #self.mat_to_flat(mat_X = self.X)
            curr_loss = self.nll_X(self.X0)
            curr_gradn = np.linalg.norm(self.grad_nll_X(res.x))
            # Store the n_iteration, losses and gradn lists        
            self._fill_tracking_lists(curr_loss, curr_gradn)

            # checking the stopping criteria
            res.x = self._check_decreasing_loss(prev_X, res.x, curr_loss, prev_loss, prev_gradn, \
                                            opt_method=opt_method,
                                            ratio_update_lr = ratio_update_lr, ftol = ftol)

            
            if self.exit_opt:
                break
                
        
        # Final PRINT
        if self.nit > 0:
            print(f"\n-Final result for {self.nit_list[-1]} its of {opt_method}")
            # NLL values
            print(f"\t-prev_loss: {prev_loss:3.8e} ~ curr_loss: {curr_loss:3.8e} --> {(prev_loss - curr_loss) / curr_loss:3.8e}")
            print(f'\t-prev_gradn: {prev_gradn:3.8e}, curr_gradn: {curr_gradn:3.8e}')

        else:
            print(f"\n-Optimization did not start -- self.nit: {self.nit}")

        # save the correct variables
        # use res.x to do all the calculations since _check_decreasing_loss may provide the prev_X stored into res.x
        self.X = self.flat_to_mat(res.x)
        self.grad = self.flat_to_mat(self.grad_nll_X(res.x))

    def adam(
            self,
            fun,
            X0,
            jac,
            iters = 1e3,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            verbose = 0,
            ftol = 1e-5,
            ratio_update_lr = 3,
            ):
            """``scipy.optimize.minimize`` compatible implementation of ADAM -
            [http://arxiv.org/pdf/1412.6980.pdf].
            Adapted from ``autograd/misc/minimizers.py``.
            """
            # copied by https://gist.github.com/jcmgray/e0ab3458a252114beecb1f4b631e19ab

            from scipy.optimize import OptimizeResult
            
            X = np.clip(X0, a_min = np.sqrt(ftol), a_max = None)
            m = np.zeros_like(X)
            v = np.zeros_like(X)

            for i in range(int(iters)):
                prev_X = X.copy()
                prev_loss = fun(prev_X)
                prev_gradn = np.linalg.norm(jac(prev_X))

                if verbose == 2:
                    print(f'-self.nll_X: {prev_loss:.15e}, gradn: {prev_gradn:.15e}',)
                
                g = jac(X)
                m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
                v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
                mhat = m / (1 - beta1**(i + 1))  # bias correction.
                vhat = v / (1 - beta2**(i + 1))
                X = X - self.learning_rate * mhat / (np.sqrt(vhat) + eps)
                X = np.clip(X, a_min = ftol, a_max = None)
                
                # STOPPING CRITERIA PART
                curr_loss = fun(X)
                X = self._check_decreasing_loss(prev_X, X, curr_loss, prev_loss, prev_gradn,
                                            opt_method="adam",
                                            ratio_update_lr = ratio_update_lr, \
                                            ftol = ftol, inner_restart = "Inner ")
                
                if self.exit_opt:
                    break

            res = OptimizeResult(x=X, fun=fun(X), jac=g, nit=i, nfev=i, success=True)

            # self.X = res.x
            # self.grad = self.flat_to_mat(res.jac[:self.n_params_X])
            self.nit += (i+1)

            return res
    
    def scipy_nll_min(self, opt_method = 'l-bfgs-b', loss_iters = 200, verbose=False,):
        
        # print the initial loss and gradn
        if verbose:
            init_jac = self.grad_nll_X(self.X0)
            print(f'\n-Inner solver initial_loss: {self.nll_X(self.X0):.8e}, gradn: {np.linalg.norm(init_jac):.8e}',)

        # fix a tolerance below which the optimization is considered converged and it will be also the lower bound for the fitnesses values
        tol = np.finfo(float).eps

        # set parameters for the optimization
        dict_options = {"ftol":tol, "gtol":tol, }
        if opt_method == "l-bfgs-b":
            dict_options.update({'maxiter':loss_iters, "disp": verbose, "maxcor": 400, "maxls": 300})
        elif opt_method == "tnc":
            dict_options.update({'maxfun':loss_iters, "disp" : verbose > 0})
        else:
            dict_options = {}

        # now OPTIMIZE!
        res = minimize(
                        self.nll_X,
                        jac = self.grad_nll_X,
                        x0 = self.X0,
                        method = opt_method,
                        bounds = [(tol,None)]*self.n_params_X,
                        options = dict_options,
                        )
        
        # keep track of the grad flow
        # self.grad = self.flat_to_mat(res.jac[:self.n_params_X])

        # # set the parameters into the final fitness
        # self.X = self.flat_to_mat(flat_X = res.x[:self.n_params_X])

        # set the number of iterations
        self.nit += res.nit
        if verbose:
            print(f'-Final res: {res}',)

        return res

    def nll(self, X, A, verbose = False):
        """Computing the Negative Log-Likelihood of the MSM model."""

        if self.name.endswith("LPCA"):
            # X = self.mat_to_flat(X)
            return self.nll_LPCA(X, tc.from_numpy(A), verbose)

        else:
            X = self.flat_to_mat(X)
            logits = X@X.T
            
            tol = np.finfo(float).eps
            triu_sum = lambda A: np.triu(A, k = 1).sum()

            nll_ver = 0
            if self.name.endswith("MSM"):
                # log1p(X) = log(1+X) = log(1+(negq)). Thus, negq \in [-1+tol, 0]
                if nll_ver == 0:
                    clip_negq = np.clip(-1-np.expm1(-logits), a_min = -1+tol, a_max = None)
                    loss_sum = triu_sum(np.where(A, -np.log1p(clip_negq), logits))
                elif nll_ver == 1:
                    pos_idx = np.where(np.triu(A, k = 1)) # np.where(np.triu(A, k = 1))
                    pos_logits = logits[pos_idx]
                    clip_pos_log = np.clip(-1-np.expm1(-pos_logits), a_min = -1+tol, a_max = None)
                    loss_sum = triu_sum(logits) - np.sum(np.log1p(clip_pos_log) + pos_logits)
                elif nll_ver == 2:
                    loss_sum = triu_sum(-A * np.log(-np.expm1(-logits)) + (1-A) * logits)

            elif self.name.endswith("CM"):
                loss_sum = triu_sum(np.log(1+logits) - A * np.log(logits))

            loss_reg = 0
            fun_l2_l1_None = lambda str_l: float(self.__dict__.get(str_l) or 0)
            if fun_l2_l1_None("reg_l2") > 0:
                loss_reg += B.pow(2).sum() / 2. * self.reg_l2
            if fun_l2_l1_None("reg_l1") > 0:
                loss_reg += np.abs(B).sum() / 2. * self.reg_l1
            
            loss = loss_sum + loss_reg
            if verbose:
                print("loss, loss_sum, loss_reg:",[tens for tens in [loss, loss_sum, loss_reg]])
            
            return loss
    
    def grad_nll(self, X, A, verbose = False):
        
        if self.name.endswith("LPCA"):
            return self.grad_nll_LPCA(X, A, verbose)

        elif self.name.endswith("MSM"):
            X = self.flat_to_mat(X)
            P = self.pmatrix_gen(X)
            R = np.reciprocal(P, where = P != 0)
            ARm1 = np.where(A, R-1, -1)

            # since we will discard the diagonal part of the nll remove the -1 * x part of the gradient
            grad = ARm1 @ X + X

            if verbose:
                print(f'-X@X.T[:4,:4]:\n {(X@X.T)[:4,:4]}',)
                print(f'-logits[:4,:4]:\n {logits[:4,:4]}',)
                print(f'-ARm1:\n {ARm1[:4,:4]}',)
                print(f'-rmv_diag[:4,:4]:\n {rmv_diag[:4,:4]}',)
                print(f'-diag[:4,:4]:\n {diag[:4]}',)
                print(f'-diag.shape: {diag.shape}',)
        
        elif self.name.endswith("CM"):
            XscX = X @ X.T
            grad = self.frmv_diag(A / XscX - self.frmv_diag(1 / (1 + XscX))) @ X
            
        return -grad.ravel()
    
    def _check_decreasing_loss(self, prev_X, curr_X, curr_loss, 
                                prev_loss, prev_gradn, opt_method, 
                                ratio_update_lr = 3, ftol = 1e-8, inner_restart = ""):
    
        rel_err_loss = (prev_loss - curr_loss) / curr_loss

        if rel_err_loss < ftol:

            if opt_method.startswith("adam"):
                
                print(f'-Stop GradDesc: loss {prev_loss:.8e}, gradn {prev_gradn:.8e}, rel_err_loss: {rel_err_loss:.2e} < {ftol:.2e}',)
                
                self.learning_rate /= ratio_update_lr
                if self.learning_rate < ftol:

                    # Exit from this iteration if learning rate sufficiently low
                    print(f'-Exiting as the new learning_rate: {self.learning_rate} < {ftol}',)
                    self.exit_opt = True
                else: 
                    print(f"\n-{inner_restart}Restart Adam @ it {self.nit_list[-1]} with new self.learning_rate: {self.learning_rate:.5}",)                
            
            else:
                print(f"\n-Break @ it {self.nit_list[-1]} due to rel_err_loss: {rel_err_loss:.2e} < {ftol:.2e}")
                self.exit_opt = True
                
            return prev_X

        return curr_X

    def _fill_tracking_lists(self, curr_loss, curr_gradn):
        self.nit_list.append(self.nit+self.nit_list[-1])
        if self.track_nll_opt:
            self.losses_list.append(curr_loss)
            self.gradn_list.append(curr_gradn)

    # solvers for LPCA

    def nll_LPCA(self, X, A, verbose = 0):
        """ Negative Log-Likelihood of the LPCA model. """
        B, C = self.flat_to_matBC(X)
        B = tc.from_numpy(B).requires_grad_(True)
        C = tc.from_numpy(C).requires_grad_(True)
        logits = B @ B.T - C @ C.T
        
        lli_LPCA = tc.nn.functional.logsigmoid(tc.where(A==1,+1,-1)*logits)
        # logits * A - tc.maximum(logits, tc.tensor([0])) - tc.log(1 + tc.exp(-tc.abs(logits))) # same as of before
        loss_sum = -tc.triu(lli_LPCA, diagonal = 1).sum()

        loss_reg = 0
        if self.get("reg_l2"):
            loss_reg += (B.pow(2).sum() + C.pow(2).sum()) / 2. * self.reg_l2
        if self.get("reg_l1"):
            loss_reg += (tc.abs(B).sum() + tc.abs(C).sum()) / 2. * self.reg_l1
        loss = loss_sum + loss_reg
        loss.backward()

        if verbose:
                print([tens.detach().numpy() for tens in [loss, loss_sum, loss_reg]])

        # self.num_grad = np.concatenate((B.grad.numpy(), C.grad.numpy()), axis = None)

        return loss.detach().numpy()

    def grad_nll_LPCA(self, X, A, verbose = False):
        B, C = self.flat_to_matBC(X)
        P = expit(B @ B.T - C @ C.T)

        diff_A_P = self.frmv_diag(A - P)
        grad_B = diff_A_P @ B
        grad_C = - diff_A_P @ C
            
        return -np.concatenate((grad_B, grad_C), axis = 1).ravel()

    def minimize_LPCA(self, A, n_epochs = None, opt_method = None, loss_iters = int(1e3), verbose=False,):
        self._set_initial_guess(obs_mat = A) # return a raveled X0
        tol = np.finfo(float).eps

        self.nll_X = lambda X: self.nll_LPCA(X, tc.tensor(A), verbose = False)
        self.grad_nll_X = lambda X: self.grad_nll_LPCA(X, A = A, verbose = False)
        
        # as in set_minimizers
        self.X = self.X0

        for i in range(n_epochs):

            self.initial_guess = self.X.copy()

            # copy the prev_ vars to stop if we will not diminishing the likelihood
            prev_X = self.X.copy()
            prev_loss = self.nll_X(prev_X)#[0]
            prev_grad = self.grad_nll_X(self.X0)
            prev_gradn = np.linalg.norm(prev_grad)

            print(f"\n-Optimization epoch {i} / {n_epochs}", )
            print(f'-Starting loss {prev_loss:.8e}, gradn: {prev_gradn:.8e}',)

            # set parameters for the optimization
            dict_options = {"ftol":tol, "gtol":tol, }
            if opt_method == "l-bfgs-b":
                dict_options.update({'maxiter': loss_iters, "disp": verbose})
            elif opt_method == "tnc":
                dict_options.update({'maxfun': loss_iters, "disp" : verbose > 0})
            else:
                dict_options = {}
        
            res = minimize(
                            self.nll_X, 
                            x0 = self.X0,
                            jac = self.grad_nll_X,
                            method = opt_method,
                            bounds = [(tol,None)]*self.n_nodes*(self.dimB+self.dimC),
                            tol = tol,
                            options = dict_options,
                            )

            curr_gradn = np.linalg.norm(res.jac)
            
            self.X = res.x
            self.X0 = res.x
            curr_loss = self.nll_X(self.X0)
            # keep track of the grad flow
            self.grad = res.jac

        print(f'-res.message: {res.message, res.nit}',)

            
        # here in pmatrix there will be computed the self.B and self.C
        self.pmatrix = self.pmatrix_gen()
        self.X = res.x.reshape(-1, self.dimB+self.dimC)
        # self.save_Xw_pmatrix()
