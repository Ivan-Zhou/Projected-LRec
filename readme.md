## Projected LRec
### Analytical Solution of the ProjectedLRec
This [notebook](Analytical%20Solution%20of%20the%20Project%20LRec.ipynb) presents the analytical solutions of the 
Projected LRec and the proof behind. 

### Functional Code
I included the code for the projected LRec in [lrec.py](lrec.py). Depending on your matrix and needs, you can choose to 
run the item-based or user-based function. 

#### Input Parameters
**The input user-item matrix is assumed to be a sparse matrix.** You can also add side attributes as an optional parameter, as well as the regularization parameter (`lam`) and the k factor. 

I included the explanation of all the input parameters in the code. They should be clear. 

#### Required Package
- numpy 1.14.0 or above
- scipy 1.0.0 or above
