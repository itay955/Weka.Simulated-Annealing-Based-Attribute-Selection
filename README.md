# Simulated Annealing Attribute Selection for Weka


###1. Reference
Kuhn and Johnson (2013), Applied Predictive Modeling, Springer\n
Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P. (1983).\n
Optimization by simulated annealing. Science, 220(4598), 671.

###2. Implemented By
Itay Hazan and Andrey Finkelstein
ISE Dept. Ben-Gurion University of the Negev, Israel

###3. Motivation for the algorithm
Several kinds of search algorithms are already embedded to Weka such as Best-First Search, Tabu Search, Hill Climbing and more. Several papers were written on Simulated Annealing but apparently no one implemented it into Weka (except for the use of Bayes network]). Therefore In our project we Implemented Simulated Annealing as one of the search algorithms 

###4. Short Description:
The main idea is that in each running iteration we are starting with a random set of features and until the average change is below a threshold we are generating a random attribute to change the current permutation. If it in the set we try to see what will happen if we remove it. If it out of the set we try to see what will happen if we will remove it. If the change enlarges the merit that we are good with the change. If the change is for the bad it is dependent with the current temperature so as long as we traverse the probability of making a bad change is reduced.  Finally after the iterations are done we are producing the subset with the best merit among all.