# Ordered Weighted Averaging sensor fusion

Sensor fusion using multiple types of Ordered Weighted Averaging (OWA).

## Steps:
1. Load recorded sensor data from "Data.xlsx", calculate MSE, MAE and RMSE.
2. Draw KDE plot and Histogram of errors.
3. Fusing all sensors data using Optimistic and Pesimistic OWA operators with parameters like [1] and calculate the errors after, and also report the amounts of Orness and Dispersion for each.
4. Test different values for &alpha; and find and report the one that produces the best result for Orness and Dispersion.
5. Do step 3 using the Induced OWA [2] Ⅾepenⅾent OWA [3] operators.








## Refrences:
[1] Filev, D., & Yager, R. R. (1998). On the issue of obtaining OWA operator weights. Fuzzy sets and systems, 94(2), 157-169. \
[2] Yager, R. R., & Filev, D. P. (1999). Induced ordered weighted averaging operators. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 29(2), 141-150. \
[3] Xu, Z. (2006, April). Dependent OWA operators. In International Conference on Modeling Decisions for Artificial Intelligence (pp. 172-178). Berlin, Heidelberg: Springer Berlin Heidelberg.