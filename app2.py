import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from PIL import Image
import graphviz



# Set the title of the app
st.title("Customer Personality Analysis")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
menu = ["Home", "Insights & Cluster Analysis", "Predict", "Visualization", "Model Development", "Model Flow"]
choice = st.sidebar.selectbox("Select a section", menu)

# Home section
if choice == "Home":
    st.subheader("Welcome to Customer Personality Analysis")
    st.write("""
        Customer Personality Analysis & Prediction
        This Customer Personality Analysis (CPA) project is an illustration of my real-time work experience
        and my portfolio. The primary objective is to leverage data analysis and machine learning techniques
        for targeted marketing strategies and enhance customer engagement.
        In the professional project, the workflow is organized into four stages:
             
            1. Customer Sentiment Analysis
             
            2. Customer Segmentation
             
            3. Predictive Model Development for Future Segmentation
             
            4. Product Recommendations
        
        The Customer Personality Analysis not only aids in understanding what customers say about products but also uncovers actionable insights based on what they do. By leveraging this dual approach, businesses can make informed decisions that foster customer loyalty and drive growth. Through clustering techniques and predictive analytics, we aim to dissect the complexities of customer behavior and tailor strategies that resonate with various customer personas.
        
        This portfolio project will focus on the second (Customer Segmentation) and third stages (Predictive Model Development), developing predictive models for future data segmentation.
    """)

# Insights & Cluster Analysis section
elif choice == "Insights & Cluster Analysis":
    st.subheader("Insights and Recommendations from Cluster Analysis")
    
    st.subheader("Overview")
    st.write("""
        The cluster analysis of customer personality has revealed two distinct groups of customers that can guide strategic marketing 
        and product development efforts. By understanding the characteristics and preferences of these clusters, businesses can tailor 
        their marketing strategies to effectively reach and engage their target audience.
    """)

    st.subheader("Customer Clusters")

    # Cluster 1
    st.write("### Cluster 1: High-Expense Customers")
    st.write("**Characteristics:**")
    st.write("""
        - Individuals with higher expenses and purchasing power.
        - Predominantly single individuals or parents with fewer than three children.
        - Generally fall into a higher income bracket.
        - Older individuals tend to be more prevalent in this cluster.
    """)
    
    st.write("**Insights:**")
    st.write("""
        This group shows a willingness to spend on premium products and services, indicating a higher level of consumer engagement 
        and brand loyalty. Their spending patterns suggest that they prioritize quality and exclusivity in their purchases, making 
        them ideal targets for luxury and high-end product lines.
    """)

    # Cluster 2
    st.write("### Cluster 2: Low-Expense Customers")
    st.write("**Characteristics:**")
    st.write("""
        - Individuals with lower expenses and purchasing power.
        - Primarily married individuals with more than three children.
        - Typically fall into a lower income bracket.
    """)
    
    st.write("**Insights:**")
    st.write("""
        This group is likely to be more budget-conscious and may prioritize essential products over luxury items. Their purchasing 
        decisions are influenced by family needs, indicating that marketing efforts should focus on affordability and value.
    """)

    st.subheader("Strategic Recommendations")
    
    # Recommendations for Cluster 1
    st.write("### Targeting Cluster 1 for Premium Products:")
    st.write("**Marketing Strategy:**")
    st.write("""
        Develop targeted marketing campaigns that emphasize exclusivity, premium quality, and unique selling propositions (USPs). 
        Use channels that resonate with this demographic, such as high-end social media platforms and premium lifestyle magazines.
    """)
    st.write("**Product Development:**")
    st.write("""
        Introduce new product lines or services that cater to luxury preferences, such as limited edition items or personalized experiences.
    """)
    st.write("**Customer Engagement:**")
    st.write("""
        Leverage loyalty programs, personalized offers, and VIP experiences to enhance engagement and retention within this segment.
    """)

    # Recommendations for Cluster 2
    st.write("### Engaging Cluster 2 with Value-Based Offerings:")
    st.write("**Marketing Strategy:**")
    st.write("""
        Create campaigns focused on value, savings, and family-oriented products. Utilize channels like family-focused online 
        platforms and community events to reach this demographic effectively.
    """)
    st.write("**Product Development:**")
    st.write("""
        Focus on affordability and practicality by developing products that meet the essential needs of larger families. 
        Consider bundled offerings that provide more value at a reduced cost.
    """)
    st.write("**Community Outreach:**")
    st.write("""
        Engage with community organizations to support local initiatives and demonstrate brand commitment to family values, 
        thereby enhancing brand loyalty among this segment.
    """)

    st.write("### Cross-Promotion Opportunities:")
    st.write("""
        Explore opportunities for cross-promotion between the two clusters. For instance, customers from Cluster 1 could be offered 
        family-oriented products from Cluster 2 that meet their lifestyle needs, while still maintaining a premium positioning. 
        Highlight benefits such as sustainable products or community support initiatives that may resonate with both segments, 
        creating a sense of shared values.
    """)

    st.write("### Continuous Monitoring and Adaptation:")
    st.write("""
        Regularly assess the evolving preferences and behaviors of both clusters through surveys and data analysis to stay aligned with their needs. 
        Adapt marketing strategies based on seasonal trends, economic changes, or shifts in consumer sentiment to ensure continued relevance.
    """)

    st.subheader("Conclusion")
    st.write("""
        By strategically leveraging the insights gained from the cluster analysis, businesses can refine their marketing efforts 
        and product offerings to effectively engage both high-spending and budget-conscious customers. Tailoring strategies to 
        meet the distinct needs of each cluster will not only enhance customer satisfaction but also drive growth and profitability.
    """)

# Predict section
elif choice == "Predict":
    st.subheader("Predictive Analysis")
    st.write("""In this section, you can predict customer responses based on their characteristics using machine learning models.
        You can input features and see the predicted response.""")

    # Load your logistic regression model
    with open('logistic.pkl', 'rb') as f:
        logistic = pickle.load(f)


    # User input for model features
    education = st.selectbox("Education Level", (0, 1, 2))  # Adjust as necessary
    marital_status = st.selectbox("Marital Status", (0, 1))  # Adjust as necessary
    income = st.number_input("Income", min_value=0.0)
    kids = st.number_input("Number of Kids", min_value=0)
    expenses = st.number_input("Expenses", min_value=0.0)
    total_accepted_cmp = st.number_input("Total Accepted Campaigns", min_value=0)
    num_total_purchases = st.number_input("Total Purchases", min_value=0)
    customer_age = st.number_input("Customer Age", min_value=0)
    customer_for = st.number_input("Customer For (duration)", min_value=0)

    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Education': [education],
        'Marital_Status': [marital_status],
        'Income': [income],
        'Kids': [kids],
        'Expenses': [expenses],
        'TotalAcceptedCmp': [total_accepted_cmp],
        'NumTotalPurchases': [num_total_purchases],
        'Customer_Age': [customer_age],
        'Customer_For': [customer_for],
    })

    if st.button("Predict Cluster"):
        # Transform the input data using PCA

        # Make predictions with the logistic regression model
        predicted_cluster = logistic.predict(input_data)

        # Display the predicted cluster
        st.success(f"The predicted customer cluster is: {predicted_cluster[0]}")

# Visualization section
# Visualization section
# Visualization section
# Visualization section
elif choice == "Visualization":
    st.header("Visualization")
    
    # List of image paths and captions
    image_paths = [
        ("C:/Users/saipr/anaconda3/Distribution_Income.png", "Distribution of Income"),
        ("C:/Users/saipr/anaconda3/Age_Distribution.png", "Age Distribution"),
        ("C:/Users/saipr/anaconda3/Expenses_Distribution.png", "Expenses Distribution"),
        ("C:/Users/saipr/anaconda3/Education_imapct.png", "Impact of Education"),
        ("C:/Users/saipr/anaconda3/Kids_Distribution.png", "Kids Distribution"),
        ("C:/Users/saipr/anaconda3/Marital_Status.png", "Marital Status"),
        ("C:/Users/saipr/anaconda3/Marital_impact.png", "Impact of Marital Status"),
        ("C:/Users/saipr/anaconda3/Days.png", "How Days Engaged impacts on Expenses?"),
        ("C:/Users/saipr/anaconda3/TotalAcc.png", "How TotalAcceptedCmp impacts on Expenses?"),
        ("C:/Users/saipr/anaconda3/kid_impact.png", "How Kids impacts on Expenses?"),  # Missing comma added here
        ("C:/Users/saipr/anaconda3/Numtotal.png", "Number of Total Purchases"),
        ("C:/Users/saipr/anaconda3/Corrilation.png", "Correlation Matrix"),
        ("C:/Users/saipr/anaconda3/Cluster_Distribution.png", "Cluster Distribution"),
        ("C:/Users/saipr/anaconda3/PCA.png", "PCA Plot"),
        ("C:/Users/saipr/anaconda3/Agg.png", "Agglomerative Clustering 1"),
        ("C:/Users/saipr/anaconda3/agg2.png", "Agglomerative Clustering 2"),
        ("C:/Users/saipr/anaconda3/Agg3.png", "Agglomerative Clustering 3"),
        ("C:/Users/saipr/anaconda3/Agg4.png", "Agglomerative Clustering 4"),
        ("C:/Users/saipr/anaconda3/confusion_matrix.png", "Confusion Matrix")
    ]

    # Set image width (you can adjust these values)
    image_width = 350  # Width of the image (in pixels)

    # Create two columns
    col1, col2 = st.columns(2)
    
    # Loop over the images and display them in two columns
    for idx, (image_path, caption) in enumerate(image_paths):
        img = Image.open(image_path)
        
        if idx % 2 == 0:  # Even index, put in the first column (col1)
            with col1:
                st.image(img, caption=caption, width=image_width)
        else:  # Odd index, put in the second column (col2)
            with col2:
                st.image(img, caption=caption, width=image_width)




elif choice == "Model Development":
    st.header("Model Development")
    st.write("Our approach to this project follows a structured, step-by-step methodology grounded in data science best practices. Each stage is thoughtfully designed to build upon the previous steps, ensuring a cohesive and comprehensive solution.")

    # Approach Overview
    st.subheader("Approach Overview")
    st.write("""
    1. **Understanding the Data**: The first step involves a thorough understanding of the dataset, its variables, and its structure. This step is crucial for shaping the subsequent stages of the project.

    2. **Data Preprocessing**: After understanding the dataset, we clean and preprocess the data. This involves handling missing values, potential outliers, and categorical variables, ensuring the data is ready for analysis.

    3. **Exploratory Data Analysis (EDA)**: This stage involves unearthing patterns, spotting anomalies, testing hypotheses, and checking assumptions through visual and quantitative methods. It provides an in-depth understanding of the variables and their interrelationships, which aids in feature selection.

    4. **Feature Selection**: Based on the insights from EDA, relevant features are selected for building the machine learning model. Feature selection is critical to improve the model's performance by eliminating irrelevant or redundant information.

    5. **Customer Segmentation**: The preprocessed data is then fed into a clustering algorithm to group customers into distinct segments based on their attributes and behavior. This segmentation enables targeted marketing and personalized customer engagement.

    6. **Model Development**: Once we have our customer segments, we develop a predictive model using a suitable machine learning algorithm. This model is trained on the current data and then validated using a separate test set.

    7. **Model Evaluation and Optimization**: The model's performance is evaluated using appropriate metrics. If necessary, the model is fine-tuned and optimized to ensure the best possible performance.

    8. **Prediction on Future Data**: The final step involves utilizing the trained model to make predictions on future data. This will allow the business to anticipate changes in customer behavior and adapt their strategies accordingly.
    """)

    st.write("""
    This approach ensures a systematic and thorough analysis of the customer data, leading to robust and reliable customer segments and predictions. It aims to provide a foundation upon which strategic business decisions can be made and future customer trends can be anticipated.
    """)

    # Continue with specific model development details
    st.subheader("Choice of Algorithms")
    st.write("""
    
    - **K-Means Clustering**: Utilized for unsupervised segmentation of customers. It partitions the data into K distinct clusters based on feature similarity.
    - **Agglomerative Clustering**: A hierarchical clustering method that recursively merges clusters based on distance metrics. It's particularly useful for identifying nested clusters in the data.
    - **Principal Component Analysis (PCA)**: A dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional form while retaining most of the variance. PCA is beneficial for visualizing complex data and improving model performance by reducing overfitting.
    - **Logistic Regression**: Used for binary classification. It helps in predicting the probability of a categorical dependent variable based on one or more predictor variables.
    - **Random Forest**: A powerful ensemble method for classification and regression tasks that operates by constructing multiple decision trees and outputting the mode of their predictions.

    """)

    # Feature Engineering
    st.subheader("Feature Engineering")
    st.write("""
    The following feature engineering steps were applied:
    - Categorical variables were encoded.
    - Numerical variables were scaled.
    - New features were derived based on domain knowledge.
    """)

    # Model Training
    st.subheader("Model Training")
    st.write("""
    The data was split into training and testing sets, followed by model training and validation using cross-validation techniques.
    """)

    # Model Evaluation
    st.subheader("Model Evaluation")
    st.write("""
    We utilized metrics such as accuracy, precision, recall, and F1-score to evaluate model performance. 
    We will present a confusion matrix and performance plots to visualize the outcomes.
    """)

    # Final Model Selection
    st.subheader("Final Model Selection")
    st.write("""
    Based on the evaluation metrics, we selected the most suitable model for deployment.
    """)

    # Visualization of Model Performance
    st.subheader("Model Performance Visualization")
    st.image("C://Users//saipr//anaconda3//confusion_matrix.png", caption="Confusion Matrix for the Selected Model")


# Function to create the flowchart for the model development process
def create_flow_chart():
    flow_chart = """
    digraph G {
        rankdir=TB;
        node [shape=box, style=filled, fillcolor="#E0E0E0"];

        start [label="Start", shape=circle, fillcolor="#FFD700"];
        understand [label="Understand the Data"];
        preprocess [label="Data Preprocessing"];
        eda [label="Exploratory Data Analysis"];
        feature_selection [label="Feature Selection"];
        customer_segmentation [label="Customer Segmentation"];
        model_development [label="Model Development"];
        evaluation [label="Model Evaluation and Optimization"];
        prediction [label="Prediction on Future Data"];
        end [label="End", shape=circle, fillcolor="#FFD700"];

        start -> understand -> preprocess -> eda -> feature_selection -> customer_segmentation -> model_development -> evaluation -> prediction -> end;
    }
    """
    return flow_chart

# Add this elif block inside your existing if-elif structur
if choice == "Model Flow":
    st.header("Model Flow")
    
    # Display the flow chart
    flow_chart = create_flow_chart()
    st.subheader("Model Flow Chart")
    st.graphviz_chart(flow_chart)

    # Add your previous model development details here
    st.write("Our approach to this project follows a structured, step-by-step methodology grounded in data science best practices...")

