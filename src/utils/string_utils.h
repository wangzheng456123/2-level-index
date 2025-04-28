#pragma once
#include <string>
#include <vector>

inline
std::string& toUpper(std::string& str) 
{
  std::transform(str.begin(), str.end(), str.begin(),
  [](char c) {
    return static_cast<char>(::toupper(c));
  });
    return str;
}

inline
bool contains(const char* str, char c) {
  for (; *str; ++str) {
    if (*str == c)
      return true;
  }
  return false;
}

inline
std::string& ltrim(std::string& str) {
  str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](char c) {
  return !std::isspace(c);
  } ));
  return str;
}

inline
std::string& rtrim(std::string& str) {
  str.erase(std::find_if(str.rbegin(), str.rend(), [](char c) {
    return !std::isspace(c);
  }).base(), str.end());
  return str;
}

inline
std::string& trim(std::string& str) {
  return ltrim(rtrim(str));
}

inline
bool isInteger(const std::string& str) {
  if (str.empty()) return false;

  size_t i = 0;
  if (str[i] == '+' || str[i] == '-') {
      i++;
  }

  for (; i < str.size(); i++) {
      if (!isdigit(str[i])) {
          return false;
      }
  }
  return i > 0;
}

inline
std::string remove_spaces(const std::string& str) {
  std::string result;
  for (char ch : str) {
    if (!isspace(ch)) {
      result += ch;
    }
  }
  return result;
}

template <typename T>
std::vector<T> parse_array(const std::string& str) {
  std::vector<T> result;
  std::string cleaned_str = str;

  if (!cleaned_str.empty() && cleaned_str.front() == '[') {
    cleaned_str.erase(0, 1); 
  }
  if (!cleaned_str.empty() && cleaned_str.back() == ']') {
    cleaned_str.pop_back(); 
  }

  std::stringstream ss(cleaned_str);
  std::string token;

  while (std::getline(ss, token, ',')) {
    token.erase(0, token.find_first_not_of(" \t"));
    token.erase(token.find_last_not_of(" \t") + 1);

    if (!token.empty()) {
      result.push_back(static_cast<T>(std::stod(token))); 
    }
  }

  return result;
}