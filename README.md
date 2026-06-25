# Movie Recommandation System
## Motivation

Recommendation systems can be designed in two forms: **collaborative filtering** and **content-based filtering**. When knowledge graphs are used alongside user ratings, the system is designed as a **hybrid** recommender system.  We use a pre-constructed knowledge graph [1] includes relationships for movies such as the **country of production**, **movie soundtrack**, and **director**. The details of these relationships are explained in the dataset section. We design a movie recommendation system based on the aformentioned knowledge graph and users' ratings of movies which provides user information, using the **Graph Attention Network (GAT)** neural architecture.

## Data Analysis
In the MovieLens-1M dataset, the numbers of users, movies, and user ratings for movies are 6,040, 3,952, and 1,000,210, respectively. User attributes include gender, age, occupation, and ZIP code. We use all of these attributes except the ZIP code for training the models.

The figures below present the distributions of user ratings for movies, as well as the distributions of users' ages and occupations. The rating distribution for men is similar to that for women, except that the number of female users is smaller. The age group 18–35 years constitutes the largest proportion of users who provide ratings.

The occupation identifiers are defined as follows:

* 0: "other" or not specified
* 1: "academic/educator"
* 2: "artist"
* 3: "clerical/admin"
* 4: "college/grad student"
* 5: "customer service"
* 6: "doctor/health care"
* 7: "executive/managerial"
* 8: "farmer"
* 9: "homemaker"
* 10: "K-12 student"
* 11: "lawyer"
* 12: "programmer"
* 13: "retired"
* 14: "sales/marketing"
* 15: "scientist"
* 16: "self-employed"
* 17: "technician/engineer"
* 18: "tradesman/craftsman"
* 19: "unemployed"
* 20: "writer"

Farmers, merchants, and unemployed individuals had the lowest participation in the surveys, while students, managers, and engineers had the highest participation.

<img width="407" height="278" alt="plot1" src="https://github.com/user-attachments/assets/14651562-c2fa-46c1-bbd3-c6e4cc84b124" />

<img width="712" height="352" alt="plot2" src="https://github.com/user-attachments/assets/78f41307-6596-4c26-9d92-647f2f26c710" />

<img width="1099" height="773" alt="plot3" src="https://github.com/user-attachments/assets/16ccba50-52d9-440a-ac2a-464b9b4de275" />

<img width="891" height="604" alt="plot4" src="https://github.com/user-attachments/assets/f8d1b80d-6561-40d5-b76d-c182a28ac725" />

<img width="2612" height="1679" alt="plot5" src="https://github.com/user-attachments/assets/328170b7-c515-4db4-931a-f49b3d580ddd" />


