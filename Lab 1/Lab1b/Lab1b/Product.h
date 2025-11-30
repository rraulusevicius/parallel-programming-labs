#ifndef PRODUCT_H
#define PRODUCT_H

#include <string>
#include <cmath>

using std::string;

struct Product {
    string name;
    int quantity;
    double price;

    // Default constructor
    Product() : name(""), quantity(0), price(0.0) {}

    // Constructor with parameters
    Product(int q, double p, const string& n)
        : name(n), quantity(q), price(p) {
    }

    // Complex calculation function 
    double calculateRiskScore() const {
        double score = 0.0;
        const int ITERATIONS = 500000;

        for (int i = 1; i <= ITERATIONS; ++i) {
            double base_calc = (quantity * price * i);

            // Multiple mathematical operations to increase complexity
            double trig_component = std::sin(base_calc / 1000.0) * std::cos(base_calc / 2000.0);
            double log_component = std::log(std::abs(base_calc) + 1) / std::log(i + 1);
            double exp_component = std::exp(-i / 50000.0) * base_calc;
            double sqrt_component = std::sqrt(std::abs(trig_component * log_component)) / i;
            double power_component = std::pow(std::abs(base_calc / i), 0.3);

            // Combine all components
            score += (trig_component + log_component + exp_component + sqrt_component + power_component) / i;

            // Add complexity every 1000 iterations
            if (i % 1000 == 0) {
                score *= 1.001;
            }

            // Extra floating point operations every 100 iterations
            if (i % 100 == 0) {
                double temp = score;
                for (int j = 0; j < 10; ++j) {
                    temp = std::sin(temp) + std::cos(temp * 0.1);
                }
                score += temp * 0.0001;
            }
        }

        // Final transformation
        double final_score = std::abs(score);

        // Add name-based component for uniqueness
        double name_hash = 0;
        for (char c : name) {
            name_hash += static_cast<double>(c) * 3.14159;
        }
        final_score += std::sin(name_hash) * 0.1;

        return final_score;
    }

    // Filter criterion: only high-risk products pass
    bool isHighRisk(double score) const {
        return score > 100000000.0;  // 100 million
    }

    // For sorting by name
    bool operator<(const Product& other) const {
        return name < other.name;
    }

    // For sorting by name
    bool operator>(const Product& other) const {
        return name < other.name;
    }


    // For equality comparison
    bool operator==(const Product& other) const {
        return name == other.name && quantity == other.quantity && price == other.price;
    }
};

// Result structure: product with its calculated score
struct ProductResult {
    Product product;
    double riskScore;

    // Default constructor (needed for std::array initialization)
    ProductResult() : product(), riskScore(0.0) {}

    // Constructor with parameters
    ProductResult(const Product& p, double score)
        : product(p), riskScore(score) {
    }

    // Sort results by product name
    bool operator<(const ProductResult& other) const {
        return product.name < other.product.name;
    }

    // > operator for comparison
    bool operator>(const ProductResult& other) const {
        return product.name > other.product.name;
    }
};

#endif // PRODUCT_H