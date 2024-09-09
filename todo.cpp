#include <bits/stdc++.h>
#include <fstream>
using namespace std;

// Bubble Sort
void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
            }
        }
    }
}

// Merge Sort
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    for (int p = 0; p < k; p++) {
        arr[left + p] = temp[p];
    }
}

void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Quick Sort
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// Standard Sort
void standardSort(vector<int>& arr) {
    sort(arr.begin(), arr.end());
}

// Algoritmo iterativo cúbico tradicional
vector<vector<int>> multiplyMatrices(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size();
    vector<vector<int>> C(n, vector<int>(n, 0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

// Algoritmo iterativo cúbico optimizado
vector<vector<int>> multiplyMatricesOptimized(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size();
    vector<vector<int>> C(n, vector<int>(n, 0));
    vector<vector<int>> B_transposed(n, vector<int>(n, 0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            B_transposed[j][i] = B[i][j];
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B_transposed[j][k];
            }
        }
    }

    return C;
}

// Funciones auxiliares para el algoritmo de Strassen
vector<vector<int>> add(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size();
    vector<vector<int>> C(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

vector<vector<int>> subtract(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size();
    vector<vector<int>> C(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

// Algoritmo de Strassen
vector<vector<int>> strassen(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size();
    if (n <= 64) {
        return multiplyMatrices(A, B);
    }

    int k = n / 2;

    vector<vector<int>> A11(k, vector<int>(k));
    vector<vector<int>> A12(k, vector<int>(k));
    vector<vector<int>> A21(k, vector<int>(k));
    vector<vector<int>> A22(k, vector<int>(k));
    vector<vector<int>> B11(k, vector<int>(k));
    vector<vector<int>> B12(k, vector<int>(k));
    vector<vector<int>> B21(k, vector<int>(k));
    vector<vector<int>> B22(k, vector<int>(k));

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + k];
            A21[i][j] = A[i + k][j];
            A22[i][j] = A[i + k][j + k];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + k];
            B21[i][j] = B[i + k][j];
            B22[i][j] = B[i + k][j + k];
        }
    }

    auto P1 = strassen(A11, subtract(B12, B22));
    auto P2 = strassen(add(A11, A12), B22);
    auto P3 = strassen(add(A21, A22), B11);
    auto P4 = strassen(A22, subtract(B21, B11));
    auto P5 = strassen(add(A11, A22), add(B11, B22));
    auto P6 = strassen(subtract(A12, A22), add(B21, B22));
    auto P7 = strassen(subtract(A11, A21), add(B11, B12));

    auto C11 = add(subtract(add(P5, P4), P2), P6);
    auto C12 = add(P1, P2);
    auto C21 = add(P3, P4);
    auto C22 = subtract(subtract(add(P5, P1), P3), P7);

    vector<vector<int>> C(n, vector<int>(n));
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            C[i][j] = C11[i][j];
            C[i][j + k] = C12[i][j];
            C[i + k][j] = C21[i][j];
            C[i + k][j + k] = C22[i][j];
        }
    }

    return C;
}

// Genera un vector aleatorio de tamaño n con valores entre min y max
vector<int> generateRandomVector(int n, int min, int max) {
    vector<int> vec(n);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(min, max);

    for (int& num : vec) {
        num = dis(gen);
    }
    return vec;
}

// Genera un vector parcialmente ordenado
vector<int> generatePartiallySortedVector(int n, double sortedPortion) {
    vector<int> vec = generateRandomVector(n, 1, 1000);
    int sortedSize = static_cast<int>(n * sortedPortion);
    sort(vec.begin(), vec.begin() + sortedSize);
    return vec;
}

// Genera un vector casi ordenado (invirtiendo algunos elementos)
vector<int> generateNearlySortedVector(int n, int swaps) {
    vector<int> vec(n);
    iota(vec.begin(), vec.end(), 0);  // Llena el vector con 0, 1, 2, ..., n-1
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, n-1);

    for (int i = 0; i < swaps; ++i) {
        int idx1 = dis(gen);
        int idx2 = dis(gen);
        swap(vec[idx1], vec[idx2]);
    }
    return vec;
}

// Genera una matriz cuadrada aleatoria de tamaño n x n con valores entre min y max
vector<vector<int>> generateRandomMatrix(int n, int min, int max) {
    vector<vector<int>> matrix(n, vector<int>(n));
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(min, max);

    for (auto& row : matrix) {
        for (int& num : row) {
            num = dis(gen);
        }
    }
    return matrix;
}

// Genera una matriz diagonal
vector<vector<int>> generateDiagonalMatrix(int n, int min, int max) {
    vector<vector<int>> matrix(n, vector<int>(n, 0));
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(min, max);

    for (int i = 0; i < n; ++i) {
        matrix[i][i] = dis(gen);
    }
    return matrix;
}

// Genera una matriz triangular superior
vector<vector<int>> generateUpperTriangularMatrix(int n, int min, int max) {
    vector<vector<int>> matrix(n, vector<int>(n, 0));
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(min, max);

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

// Función para guardar un vector en un archivo
void saveVectorToFile(const vector<int>& vec, const string& filename) {
    ofstream file(filename);
    for (int num : vec) {
        file << num << " ";
    }
    file.close();
}

// Función para guardar una matriz en un archivo
void saveMatrixToFile(const vector<vector<int>>& matrix, const string& filename) {
    ofstream file(filename);
    for (const auto& row : matrix) {
        for (int num : row) {
            file << num << " ";
        }
        file << "\n";
    }
    file.close();
}

// Función para guardar resultados en un archivo
void saveResultsToFile(const string& results, const string& filename) {
    ofstream file(filename, ios::app);
    file << results;
    file.close();
}

void generateAndSaveDatasets() {
    vector<int> sizes = {100, 1000, 10000, 100000};
    
    for (int size : sizes) {
        auto randomVec = generateRandomVector(size, 1, 1000);
        auto partiallySortedVec = generatePartiallySortedVector(size, 0.5);
        auto nearlySortedVec = generateNearlySortedVector(size, size/10);
        
        saveVectorToFile(randomVec, "random_vector_" + to_string(size) + ".txt");
        saveVectorToFile(partiallySortedVec, "partially_sorted_vector_" + to_string(size) + ".txt");
        saveVectorToFile(nearlySortedVec, "nearly_sorted_vector_" + to_string(size) + ".txt");
        
        int matrixSize = min(size, 1000);
        auto randomMatrix1 = generateRandomMatrix(matrixSize, 1, 100);
        auto randomMatrix2 = generateRandomMatrix(matrixSize, 1, 100);
        auto diagonalMatrix = generateDiagonalMatrix(matrixSize, 1, 100);
        auto triangularMatrix = generateUpperTriangularMatrix(matrixSize, 1, 100);
        
        saveMatrixToFile(randomMatrix1, "random_matrix1_" + to_string(matrixSize) + ".txt");
        saveMatrixToFile(randomMatrix2, "random_matrix2_" + to_string(matrixSize) + ".txt");
        saveMatrixToFile(diagonalMatrix, "diagonal_matrix_" + to_string(matrixSize) + ".txt");
        saveMatrixToFile(triangularMatrix, "triangular_matrix_" + to_string(matrixSize) + ".txt");
    }
}

// Función para medir el tiempo de ejecución de una función
template<typename Func, typename... Args>
long long measureTime(Func func, Args&&... args) {
    auto start = chrono::high_resolution_clock::now();
    func(forward<Args>(args)...);
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::microseconds>(end - start).count();
}

// Función para ejecutar y medir algoritmos de ordenamiento
string measureSortingAlgorithms(vector<int>& arr) {
    vector<int> arr_copy = arr;
    stringstream results;
    
    results << "Array size: " << arr.size() << "\n";
    
    long long bubbleTime = measureTime(bubbleSort, arr_copy);
    results << "Bubble Sort time: " << bubbleTime << " microseconds\n";
    
    arr_copy = arr;
    long long mergeTime = measureTime([&arr_copy]() { mergeSort(arr_copy, 0, arr_copy.size() - 1); });
    results << "Merge Sort time: " << mergeTime << " microseconds\n";
    
    arr_copy = arr;
    long long quickTime = measureTime([&arr_copy]() { quickSort(arr_copy, 0, arr_copy.size() - 1); });
    results << "Quick Sort time: " << quickTime << " microseconds\n";
    
    arr_copy = arr;
    long long stdTime = measureTime(standardSort, arr_copy);
    results << "Standard Sort time: " << stdTime << " microseconds\n\n";
    
    return results.str();
}

// Función para ejecutar y medir algoritmos de multiplicación de matrices
string measureMatrixMultiplication(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    stringstream results;
    results << "Matrix size: " << A.size() << "x" << A.size() << "\n";
    
    long long traditionalTime = measureTime(multiplyMatrices, A, B);
    results << "Traditional multiplication time: " << traditionalTime << " microseconds\n";
    
    long long optimizedTime = measureTime(multiplyMatricesOptimized, A, B);
    results << "Optimized multiplication time: " << optimizedTime << " microseconds\n";
    
    long long strassenTime = measureTime(strassen, A, B);
    results << "Strassen multiplication time: " << strassenTime << " microseconds\n\n";
    
    return results.str();
}

int main() {
    generateAndSaveDatasets();
    
    vector<int> sizes = {100, 1000, 10000, 100000};
    
    for (int size : sizes) {
        // Medir algoritmos de ordenamiento
        vector<int> randomVec = generateRandomVector(size, 1, 1000);
        vector<int> partiallySortedVec = generatePartiallySortedVector(size, 0.5);
        vector<int> nearlySortedVec = generateNearlySortedVector(size, size/10);
        
        string results = "Random Vector:\n";
        results += measureSortingAlgorithms(randomVec);
        
        results += "Partially Sorted Vector:\n";
        results += measureSortingAlgorithms(partiallySortedVec);
        
        results += "Nearly Sorted Vector:\n";
        results += measureSortingAlgorithms(nearlySortedVec);
        
        // Medir algoritmos de multiplicación de matrices
        int matrixSize = min(size, 1000);
        vector<vector<int>> randomMatrix1 = generateRandomMatrix(matrixSize, 1, 100);
        vector<vector<int>> randomMatrix2 = generateRandomMatrix(matrixSize, 1, 100);
        
        results += "Random Matrices:\n";
        results += measureMatrixMultiplication(randomMatrix1, randomMatrix2);
        
        vector<vector<int>> diagonalMatrix = generateDiagonalMatrix(matrixSize, 1, 100);
        results += "Random Matrix * Diagonal Matrix:\n";
        results += measureMatrixMultiplication(randomMatrix1, diagonalMatrix);
        
        vector<vector<int>> triangularMatrix = generateUpperTriangularMatrix(matrixSize, 1, 100);
        results += "Random Matrix * Upper Triangular Matrix:\n";
        results += measureMatrixMultiplication(randomMatrix1, triangularMatrix);
        
        saveResultsToFile(results, "results.txt");
    }
    
    return 0;
}