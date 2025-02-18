using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using ExpenseTrackerAPI.Data;
using ExpenseTrackerAPI.Models;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace ExpenseTrackerAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ExpensesController : ControllerBase
    {
        private readonly ExpenseContext _context;
        private readonly HttpClient _httpClient;

        public ExpensesController(ExpenseContext context, HttpClient httpClient)
        {
            _context = context;
            _httpClient = httpClient;
        }

        // ✅ Get all expenses
        [HttpGet]
        public async Task<ActionResult<IEnumerable<Expense>>> GetExpenses()
        {
            return await _context.Expenses.ToListAsync();
        }

        // ✅ Add a new expense (with AI categorization)
        [HttpPost]
        public async Task<IActionResult> PostExpense(Expense expense)
        {
            if (expense == null || string.IsNullOrEmpty(expense.Description) || expense.Amount <= 0)
            {
                return BadRequest("Invalid expense data.");
            }

            // ✅ If no category is provided, call AI API to predict one
            if (string.IsNullOrEmpty(expense.Category))
            {
                var requestBody = new { description = expense.Description };
                var content = new StringContent(JsonSerializer.Serialize(requestBody), Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync("http://localhost:5000/predict", content);
                if (response.IsSuccessStatusCode)
                {
                    var responseBody = await response.Content.ReadAsStringAsync();
                    var prediction = JsonSerializer.Deserialize<PredictionResponse>(responseBody);
                    expense.Category = prediction?.Category ?? "Uncategorized";  // ✅ Default category if null
                }
                else
                {
                    expense.Category = "Uncategorized";  // ✅ Fallback category if AI API fails
                }
            }

            _context.Expenses.Add(expense);
            await _context.SaveChangesAsync();

            return CreatedAtAction(nameof(GetExpenses), new { id = expense.Id }, expense);
        }

        // ✅ AI Prediction Response Model
        private class PredictionResponse
        {
            public string Category { get; set; } = "Uncategorized";  // ✅ Default value assigned
        }
    }
}
