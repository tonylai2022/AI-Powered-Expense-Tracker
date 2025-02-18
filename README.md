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

### 1. Clone the Repository
```bash
git clone https://github.com/tonylai2022/AI-Powered-Expense-Tracker.git
cd AI-Powered-Expense-Tracker
