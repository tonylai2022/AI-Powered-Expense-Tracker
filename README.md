# AI-Powered Expense Tracker

This project is an **AI-powered expense tracker** that leverages machine learning models (TensorFlow and PyTorch) for categorizing expenses. It allows users to record their expenses, view predictions based on the description of each expense, and track them over time. The backend integrates with an AI model that classifies expenses into predefined categories.

## Features

- **AI-Powered Categorization:** Uses TensorFlow and PyTorch models to categorize expenses automatically.
- **Expense Management:** Add, view, and categorize expenses.
- **Frontend (Blazor Web App):** User-friendly interface built with Blazor for viewing and adding expenses.
- **Model Selection:** Option to choose either TensorFlow or PyTorch for predictions.
- **Backend (ASP.NET Core API):** RESTful API for managing expenses and interacting with AI models.
- **Database:** Uses Entity Framework and SQL Server for persistent data storage.

## Tech Stack

- **Frontend:** 
  - Blazor WebAssembly
  - HTML, CSS
- **Backend:**
  - ASP.NET Core Web API
  - C#
  - Entity Framework Core
  - HttpClient to interact with AI models
- **AI Models:**
  - TensorFlow (Python)
  - PyTorch (Python)
- **Database:** 
  - SQL Server (with Entity Framework Core)

## Prerequisites

Before you begin, ensure you have met the following requirements:
- **.NET SDK 6.0 or later** installed: [Download .NET SDK](https://dotnet.microsoft.com/download)
- **Python 3.8+** installed: [Download Python](https://www.python.org/downloads/)
- **TensorFlow** and **PyTorch** installed for AI prediction: 
    ```bash
    pip install tensorflow torch
    ```
- **SQL Server** or any compatible database engine for running the application.

## Setup Instructions

## 1. Clone the Repository

git clone https://github.com/tonylai2022/AI-Powered-Expense-Tracker.git
cd AI-Powered-Expense-Tracker

## 2. Backend Setup

1. Open the `ExpenseTrackerAPI` directory in Visual Studio or your preferred IDE.
2. Install the required NuGet packages:
    - **Microsoft.EntityFrameworkCore.SqlServer**
    - **Microsoft.EntityFrameworkCore.Tools**
    - **Microsoft.AspNetCore.HttpClientFactory**

3. Set up the database:
    - Modify the `appsettings.json` file to provide your SQL Server connection string.
    - Apply migrations by running the following command in the package manager console:
      ```bash
      dotnet ef database update
      ```

4. Run the backend API:
    - In the `ExpenseTrackerAPI` directory, run:
      ```bash
      dotnet run
      ```

---

## 3. Frontend Setup

1. Open the Blazor WebAssembly project (`ExpenseTrackerUI`) in Visual Studio.
2. Ensure the Blazor app is set to call the backend API for expense tracking and model prediction.
3. Run the Blazor app:
    ```bash
    dotnet run
    ```

---

## 4. AI Model Setup

1. Ensure the **TensorFlow** and **PyTorch** models are set up correctly:
    - Train the models on your data using Python (instructions available in `train_model.py`).
    - Save the models as `expense_classifier_tf.h5` (for TensorFlow) and `expense_classifier_pytorch.pth` (for PyTorch).

2. Run the AI models' prediction service:
    - Start the Python API for both TensorFlow and PyTorch. These models should listen to predictions at `http://localhost:5000/predict`.

---

## 5. Testing the Application

1. Open the Blazor application in your browser.
2. Use the form to add a new expense and select the model for predictions (either TensorFlow or PyTorch).
3. View the categorized expense and ensure predictions are displayed and saved in the database.

---

## 6. Deploying to Production

- To deploy the application to a production environment, ensure that you configure the backend API and database correctly.
- Host the Blazor application and the ASP.NET Core API on a suitable web server or cloud provider like **Azure** or **AWS**.
- Make sure the models are hosted or available in a production environment for predictions.

