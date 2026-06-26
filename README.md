# Movie Recommandation System
## Motivation

Recommendation systems can be designed in two forms: **collaborative filtering** and **content-based filtering**. When knowledge graphs are used alongside user ratings, the system is designed as a **hybrid** recommender system.  We use a pre-constructed knowledge graph [1] includes relationships for movies such as the **country of production**, **movie soundtrack**, and **director**. I design a movie recommendation system based on the aformentioned knowledge graph and the MovieLens dataset [2] which provides user information, using the **Graph Attention Network (GAT)** neural architecture.

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

<img width="549.5" height="386.5" alt="plot3" src="https://github.com/user-attachments/assets/16ccba50-52d9-440a-ac2a-464b9b4de275" />

<img width="891" height="604" alt="plot4" src="https://github.com/user-attachments/assets/f8d1b80d-6561-40d5-b76d-c182a28ac725" />

<img width="2612" height="1679" alt="plot5" src="https://github.com/user-attachments/assets/328170b7-c515-4db4-931a-f49b3d580ddd" />

### Knowlege Graph sampled items
The knowlege graph items are provided in the form of (movie_id, relation, movie_id). Here are a few examples:
```
11904 film.film.producer 16954
348 film.film.actor 16955
13598 film.film.costume_designer 16956
9098 film.film.actor 16957
```

## Related Work and Our Contribution

Among previous works on the MovieLens-1M dataset, matrix completion methods have achieved strong performance [3,4,5]. In [5], graph neural networks were used for matrix completion. Another category of models relies only on graph neural networks and attempts to design a hybrid recommendation system [1,6]. According to my study, The error of the model proposed in [6] is higher than the proposed model in [1], called KGCN. Therefore, We use KGCN for our contribution to make an improvement. They trained the KGCN model on knowledge graphs for the MovieLens-20M dataset. So, We filtered the knowledge graph to be compatible with the MovieLens-1M dataset. We make two contributions:

* We design KGAT based on Graph Attention Networks (GAT) instead of Graph Convolutional Networks (GCN) for learning from the knowledge graph. In attention-based graph neural networks, the model is able to learn the relative importance of neighboring nodes by assigning different weights to them.

*  The KGCN model is allowed to learn the initial user features and construct a representation for each user from random input embeddings [2]. Instead, We utilize user metadata and construct initial feature vectors for each user based on the available user information.

## References

[1] Wang, Hongwei, et al. "Knowledge graph convolutional networks for recommender systems." The world wide web conference. 2019. The knowledge graph is available at https://github.com/zzaebok/KGCN-pytorch 

[2] Harper, F. Maxwell, and Joseph A. Konstan. "The movielens datasets: History and context." Acm transactions on interactive intelligent systems. 2015.

[3] Han, Soyeon Caren, et al. "Glocal-k: Global and local kernels for recommender systems." Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021.

[4] Shen, Wei, et al. "Inductive matrix completion using graph autoencoder." Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021.

[5] Zhang, Muhan, and Yixin Chen. "Inductive matrix completion based on graph neural networks." arXiv preprint arXiv:1904.12058 (2019).

[6] Darban, Zahra Zamanzadeh, and Mohammad Hadi Valipour. "GHRS: Graph-based hybrid recommendation system with application to movie recommendation." Expert Systems with Applications. 2022.

