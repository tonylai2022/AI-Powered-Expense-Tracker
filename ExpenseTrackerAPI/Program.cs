using Microsoft.EntityFrameworkCore;
using ExpenseTrackerAPI.Data;

var builder = WebApplication.CreateBuilder(args);

// ✅ Register Database Context (Fix Dependency Injection issue)
builder.Services.AddDbContext<ExpenseContext>(options =>
    options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection"))); 

// ✅ Register HttpClient (Fixes HttpClient dependency issue in ExpensesController)
builder.Services.AddHttpClient();

// ✅ Add CORS Policy
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowBlazorUI",
        policy => policy.WithOrigins("http://localhost:5131") // Blazor UI URL
                        .AllowAnyMethod()
                        .AllowAnyHeader());
});

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// ✅ Use CORS Before Authorization
app.UseCors("AllowBlazorUI");

app.UseHttpsRedirection();
app.UseAuthorization();
app.MapControllers();
app.Run();
