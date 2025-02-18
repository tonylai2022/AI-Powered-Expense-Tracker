using System;

namespace ExpenseTrackerAPI.Models
{
    public class Expense
    {
        public int Id { get; set; }
        public string Description { get; set; } = string.Empty; // ✅ Fix: Provide default value
        public decimal Amount { get; set; }
        public string? Category { get; set; } // ✅ Made nullable
        public DateTime DateAdded { get; set; } = DateTime.UtcNow;  // ✅ Fix: Default date

        // ✅ Fields for AI model predictions
        public string TensorFlowCategory { get; set; } = string.Empty;  // ✅ Default empty string
        public string PyTorchCategory { get; set; } = string.Empty;     // ✅ Default empty string
    }
}
