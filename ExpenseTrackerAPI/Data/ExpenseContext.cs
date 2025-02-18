using Microsoft.EntityFrameworkCore;
using ExpenseTrackerAPI.Models; // Ensure the namespace is correct

namespace ExpenseTrackerAPI.Data; // Ensure this namespace is correct

public class ExpenseContext : DbContext
{
    public ExpenseContext(DbContextOptions<ExpenseContext> options) : base(options) { }

    public DbSet<Expense> Expenses { get; set; }
}
