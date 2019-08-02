# Monte-Carlo-Tree-Search-
Implementation of a basic Simulator and a Monte-Carlo Tree Search Agent that produces a farm plan that maximizes the food produced by the farm.  
## Following are the assumptions made while designing the simulator
1. The yield of the plant X increases even if the plant Y, which helps plant X, was planted adjacent to X just one day before harvesting X. So it considers presence of helpful plants adjacent to the plant being harvested only during the time of harvesting the plants.
2. As both the plants take 90 days to grow and we have to give a plan that gives maximum yield at the end of 91 days the best approach would be to plant all the plants on day 1 and so simulator is not considering any parameter to keep note of time.
3. Thus, the total yield returned by the simulator is the yield that can be obtained at the end of the 90 days period if the given plan is followed and all the plants are planted on the first day.
