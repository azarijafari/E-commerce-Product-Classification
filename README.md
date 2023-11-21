# E-commerce Product Classification
## 1. Introduction

In today's digital landscape, data mining and machine learning techniques are employed to enhance the performance and user experience of online stores. Data mining is utilized to analyze diverse data related to products, customer needs, and past purchasing behaviors, aiming to identify hidden patterns and correlations within the data. This data analysis through mining contributes to the improvement of marketing strategies and effective decision-making for content management and product categorization.

### 1.1. Project Overview
This project is designed as an educational initiative, focusing on the implementation of machine learning techniques to build an intelligent classification system based on product features and user feedback. These techniques include the use of classification algorithms such as Logistic Regression, Gaussian Naive Bayes, Random Forest, Decision Tree, K-Nearest Neighbors, and Support Vector Machines (SVM). By training on existing data, these algorithms can automatically categorize products into different classes, enhancing the overall shopping experience. This project plays a crucial role in increasing classification accuracy and boosting the store's capabilities in content generation and systematic product uploading on the e-commerce website.

To advance the project objectives, we have designed a pipeline, the overall schema of which is depicted in the figure below.
……………………………….

### 1.2. Research Structure
The rest of this report is organized as follows
The structure of this research is outlined below and will be further elaborated in the subsequent sections.
```
Project Stracture                               
    ├── 2. Data Management                 
    │   ├── 2.1. Initial Data         
    │   ├── 2.2. Data Integration
    │   ├── 2.3. Data Preprocessing	
    │   └── 2.4. Data Preparation for model training                              
    │	   ├── 2.4.1. Category Column          
    │      ├── 2.4.2. Rating Column
    │      └── 2.4.3. Review-Text, Product_Title Columns
    │	      ├── Handling Outlier Data          
    │         ├── Handling Out of Vector Words 
    │         ├── Building Word Vectors	
    │         └── Data Spliting
    ├── 3. Model Training and Evaluation         
    │   ├── 3.1. Model Tuning by Cross Validation
    │   ├── 3.2. Fitting Model on Train Data with Best Hyperparameters	
    │   └── 3.3. Model Evaluation                             
    ├── 4. Models' Comparison                                    
    └── 5. Conclusion
```

## 2. Data Management
In this section, we systematically process initial data to obtain suitable data for training models.
### 2.1. Initial dataset
The data for this project pertains to an online store that offers products in two categories: kitchen and jewelry. The extracted dataset comprises 1754 samples stored in two types of files, each linked through the "id" column. These two types of files are as follows:

1. Four files containing product information (id, category, product_title).
2. Four files containing user feedback (id, rating, review_text).

Each of the features in these two file types includes the following information:

1. `Id`: A unique identifier for each product.
2. `Category`: Including two categories of products, "Kitchen" or "Jewelry."
3. `Product_Title`: A brief description of the product.
4. `Rating`: The rating given to the product by users.
5. `Review_Text`: Users' comments and feedback about the product.

### 2.2. Data Integration

In this stage, we combine individual datasets and create a final dataset for use in subsequent steps. The initial dataset consists of four files containing product information and four files containing user feedback, as described below.

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" colspan="2">   <br>Product   information   </th>
    <th class="tg-0pky" colspan="2">   <br>User’s   Feedback   </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">   <br>Product-data-0   </td>
    <td class="tg-0pky">   <br>500   </td>
    <td class="tg-0pky">   <br>Review-0   </td>
    <td class="tg-0pky">   <br>500   </td>
  </tr>
  <tr>
    <td class="tg-0pky">   <br>Product-data-1   </td>
    <td class="tg-0pky">   <br>500   </td>
    <td class="tg-0pky">   <br>Review-1   </td>
    <td class="tg-0pky">   <br>500   </td>
  </tr>
  <tr>
    <td class="tg-0pky">   <br>Product-data-2   </td>
    <td class="tg-0pky">   <br>500   </td>
    <td class="tg-0pky">   <br>Review-2   </td>
    <td class="tg-0pky">   <br>500   </td>
  </tr>
  <tr>
    <td class="tg-0pky">   <br>Product-data-3   </td>
    <td class="tg-0pky">   <br>254   </td>
    <td class="tg-0pky">   <br>Review-3   </td>
    <td class="tg-0pky">   <br>254   </td>
  </tr>
</tbody>
</table>


In the process of combining these datasets, two issues were identified:

1. In the "review-2" file, the order of the two columns, "rating" and "id," was incorrect. This problem was resolved by swapping the positions of these two columns.
   
2. In the "product-data-4" file, the word "Kitchen" in the "category" column was mistakenly spelled as "ktchen," which has been corrected.

Following these adjustments, we combined the four "product-data" files into one dataset and the four "review" files into another dataset. Subsequently, using the unique "id" column in each row, we merged the two resulting datasets, creating a final consolidated dataset. Below, you can find a preview of five rows from the final dataset.
…………………………………

The "category" column, considered as the target for prediction in this project, indicates that the resulting dataset comprises 870 samples of "Jewelry" and 884 samples of "Kitchen." This demonstrates a balanced distribution in the dataset.
………………………………

The "rating" column, representing the user-assigned rating to each product, which includes one of the numbers {1, 2, 3, 4, 5}.
…………………………….
### 2.3. Data Preprocessing

In this stage, we clean the text within the "product_title" and "review_text" columns to prepare it for the next step, which involves data preparation for model training. The techniques employed in this section include:

- Text lowercase
- Cleaning the text from emails, symbols, and web links
- Removing numbers
- Cleaning punctuation marks
- Lemmatization

You can see an example of this data cleaning process below.

For the "product_title" column:
Before cleaning: Silver-tone Cross With Caduceus Emblem Nurse Pen - Perfect Nurse Gift
After Cleaning: silver tone cross with caduceus emblem nurse pen perfect nurse gift

For the "review_text" column:
Before cleaning: Excellent pan - my daughter just made madeleines - yum!
After cleaning: excellent pan my daughter just made madeleines yum


### 2.4. Data Preparation for Model Training

In this stage, we need to transform the data into a numerical format suitable for machine learning models. The following methods are applied:

#### 2.4.1. "Category" Column
Instead of "Jewelry," we assign the number 0, and for "Kitchen," we assign the number 1.

#### 2.4.2. "Rating" Column
Normalizing the numbers in this column in a consistent manner. Instead of the numbers [1, 2, 3, 4, 5], we use the respective values [-1, -0.5, 0, 0.5, 1]. Since the rating range remains constant and does not change, this fixed normalization method helps prevent data leakage from the test set during model training. It ensures that the numbers are within a comprehensible range for the model and are presented in the scale of word vectors.

#### 2.4.3. "product_title" and "review_text" Columns
These two columns contain text, and the texts have varying token counts. Here, we concatenate these two columns for each text sample with a space in between and tokenize them. The resulting texts have a minimum token count of 5 and a maximum token count of 623. The distribution of tokens in the texts can be observed in the figure below.
…………………………………………..

**- Handling Outlier Data**

In this figure, it is evident that we encounter some outlier data points with a higher token count than the mode of the samples. An important method in text processing to handle outlier data is the use of the Bag of Words (BOW) technique. Instead of employing embeddings for individual words, BOW calculates an embedding vector for the entire text, usually by taking the average of the embedding vectors for the words in that text. Given that machine learning methods are employed in this project and a consistent vector is required for each sample, we utilize this approach.

**- Handling Out of Vector Words**

The embedding method employed in this research is fastText, developed by Meta. This method has shown promising results in various studies, allowing us to determine the embedding length as a hyperparameter, considering it for model tuning. Additionally, this method provides the ability to represent out-of-vector (oov) words, for which there is no default embedding, by generating suitable embeddings using bigrams of that word. This capability is crucial in covering new words to provide a better representation of the entire sentence to the model.

**- Building Word Vectors**

For each text, we first obtain the embedding for each token using the fastText method and consider the average of the embeddings for each text as the text representation. It's worth noting that we experimented with several embedding lengths and ultimately settled on a length of 64. Afterward, we extract the normalized rating for each sample from the "rating" column and concatenate it to the end of this vector. In the end, for each sample from the final data, we have a numerical vector of 65 dimensions and a category of 0 or 1.

**- Data Splitting**

Given that the total number of samples is 1754, we end up with a matrix (1754 * 65) for X_data and a vector (1 * 1754) for y_data. Afterward, we randomly split these data into two parts at a ratio of 80% and 20%: (X_train, y_train) with 1403 samples and (X_test, y_test) with 351 samples. It is noteworthy that in this research, we utilize the cross-validation method for model tuning, which is highly suitable for cases with limited data. The class distribution for the two parts of data is provided in the table below.
………………………………….

## 3. Model Training and Evaluation

In this research, to predict the category based on the generated representations, we employed six machine learning models. The utilized models include Logistic Regression, Gaussian Naive Bayes, Random Forest, Decision Tree, K-Nearest Neighbors, and Support Vector Machines (SVM). The following steps were taken:

### 3.1. Model Tuning with Cross-Validation
### 3.2. Fitting Model on Train Data with Best Hyperparameters
### 3.3. Model Evaluation on Test Data

Given the limited number of samples and to enhance model training quality, we used the Cross-Validation method for model tuning. Considering that our data is balanced in two classes, we utilized accuracy as the evaluation metric for cross-validation. The table below reports the evaluation results of cross-validation and the final model testing. The best result belongs to the Logistic Regression method, achieving an accuracy of 96.30%.
…………………………….
The confusion matrices related to the evaluation results of these models are provided in the figures below. The high performance of the Logistic Regression method is evident in both classes.
…………………………….
## 4. Model Comparison
In the following plots, the six employed models are compared based on evaluation metrics such as Accuracy, Precision, Recall, and F1-Score. These visuals provide an overview of the results obtained from this research.
……………………………
## 5. Conclusion
