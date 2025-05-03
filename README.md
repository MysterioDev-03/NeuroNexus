# Titanic Survival Prediction Web App

This project is a web application built using Flask, HTML, and CSS to predict survival chances of passengers aboard the Titanic. The app allows users to input various attributes such as age, sex, class, and other details, and predicts the likelihood of survival using a machine learning model.

## Features

- Input passenger details such as age, sex, class, and more.
- Predict the survival probability based on a trained machine learning model.
- Display the results in a user-friendly interface.
- Built with Flask for the backend and HTML/CSS for the frontend.

## Installation

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/your-username/titanic-survival-prediction.git
    ```

2. Navigate into the project directory:

    ```bash
    cd titanic-survival-prediction
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. After installation, run the Flask application:

    ```bash
    python app.py
    ```

2. Open your browser and go to `http://127.0.0.1:5000/` to interact with the Titanic survival prediction web app.

3. Input the necessary details for a passenger, and the app will display the survival prediction.

## Technologies Used

- **Flask**: Web framework for the backend.
- **HTML**: Structure of the web pages.
- **CSS**: Styling the frontend.
- **Scikit-learn**: For the machine learning model used to predict survival.
- **Pandas**: For data manipulation.
- **NumPy**: For numerical operations.

## Machine Learning Model

The model used in this app is trained using the Titanic dataset from Kaggle. It uses a combination of features such as:

- Passenger's age
- Sex (male/female)
- Passenger class
- Embarked location
- SibSp (siblings/spouses aboard)
- Parch (parents/children aboard)

The machine learning model is built with **Scikit-learn**, using an appropriate classification algorithm (e.g., Random Forest, Logistic Regression).

## Contributing

Feel free to fork this repository and submit pull requests if you have suggestions for improvements or bug fixes. All contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Titanic dataset from Kaggle: https://www.kaggle.com/c/titanic
- Flask documentation: https://flask.palletsprojects.com/
