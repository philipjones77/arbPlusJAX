#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include <cmath>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string extract_string_field(const std::string& text, const std::string& key) {
  const std::regex rx("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  if (!std::regex_search(text, match, rx)) {
    throw std::runtime_error("missing string field: " + key);
  }
  return match[1].str();
}

std::vector<double> extract_number_array(const std::string& text, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const auto key_pos = text.find(needle);
  if (key_pos == std::string::npos) {
    return {};
  }
  const auto open = text.find('[', key_pos);
  const auto close = text.find(']', open);
  if (open == std::string::npos || close == std::string::npos || close < open) {
    throw std::runtime_error("malformed array field: " + key);
  }
  const std::string body = text.substr(open + 1, close - open - 1);
  const std::regex num_rx(R"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)");
  std::vector<double> out;
  for (std::sregex_iterator it(body.begin(), body.end(), num_rx), end; it != end; ++it) {
    out.push_back(std::stod(it->str()));
  }
  return out;
}

double eval_unary(const std::string& name, double x) {
  if (name == "exp") return std::exp(x);
  if (name == "log") return std::log(x);
  if (name == "sqrt") return std::sqrt(x);
  if (name == "sin") return std::sin(x);
  if (name == "cos") return std::cos(x);
  if (name == "tan") return std::tan(x);
  if (name == "sinh") return std::sinh(x);
  if (name == "cosh") return std::cosh(x);
  if (name == "tanh") return std::tanh(x);
  if (name == "gamma") return boost::math::tgamma(x);
  if (name == "erf") return boost::math::erf(x);
  if (name == "erfc") return boost::math::erfc(x);
  throw std::runtime_error("unsupported unary function: " + name);
}

double eval_bivariate(const std::string& name, double nu, double z) {
  if (name == "besselj") return boost::math::cyl_bessel_j(nu, z);
  if (name == "bessely") return boost::math::cyl_neumann(nu, z);
  if (name == "besseli") return boost::math::cyl_bessel_i(nu, z);
  if (name == "besselk") return boost::math::cyl_bessel_k(nu, z);
  throw std::runtime_error("unsupported bivariate function: " + name);
}

void write_json_array(const std::vector<double>& ys) {
  std::cout << '[';
  for (std::size_t i = 0; i < ys.size(); ++i) {
    if (i) std::cout << ',';
    std::ostringstream oss;
    oss.precision(17);
    oss << ys[i];
    std::cout << oss.str();
  }
  std::cout << ']';
}

}  // namespace

int main() {
  try {
    const std::string input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    const std::string fn_name = extract_string_field(input, "function");
    const std::vector<double> xs = extract_number_array(input, "x");
    const std::vector<double> nus = extract_number_array(input, "nu");
    const std::vector<double> zs = extract_number_array(input, "z");

    std::vector<double> ys;
    if (!nus.empty() || !zs.empty()) {
      if (nus.size() != zs.size()) {
        throw std::runtime_error("nu/z length mismatch");
      }
      ys.reserve(nus.size());
      for (std::size_t i = 0; i < nus.size(); ++i) {
        ys.push_back(eval_bivariate(fn_name, nus[i], zs[i]));
      }
    } else {
      ys.reserve(xs.size());
      for (double x : xs) {
        ys.push_back(eval_unary(fn_name, x));
      }
    }

    write_json_array(ys);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << '\n';
    return 2;
  }
}
