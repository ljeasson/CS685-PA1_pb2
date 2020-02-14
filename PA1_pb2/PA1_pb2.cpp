#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#include "ReadImage.cpp"
#include "WriteImage.cpp"

#define pi 3.141592653589793

using namespace cv;
using namespace std;

void generate_gaussian_kernel_2D(float** kernel, double sigma, int kernel_size);
void padd_with_zeros_2D(int** matrix, int** padded_matrix, int width, int height, int filter_size);
void apply_sobel(int** image, int x_size, int y_size, int kernel[3][3], int kernel_size, int** output_image);


void generate_gaussian_kernel_2D(float** kernel, double sigma, int kernel_size)
{
    int i, j;
    float cst, tssq, x, sum;

    cst = 1. / (sigma * sqrt(2.0 * pi));
    tssq = 1. / (2 * sigma * sigma);

    for (i = 0; i < kernel_size; i++)
    {
        for (j = 0; j < kernel_size; j++)
        {
            x = (float)(i - kernel_size / 2);
            kernel[i][j] = (cst * exp(-(x * x * tssq)));
        }
    }

    sum = 0.0;
    for (i = 0; i < kernel_size; i++)
        for (j = 0; j < kernel_size; j++)
            sum += kernel[i][j];

    for (i = 0; i < kernel_size; i++)
        for (j = 0; j < kernel_size; j++)
            kernel[i][j] /= sum;
}

void padd_with_zeros_2D(int** matrix, int** padded_matrix, int width, int height, int filter_size)
{
    int new_height = height + filter_size - 1;
    int new_width = width + filter_size - 1;

    for (int i = 0; i < new_height; i++)
        for (int j = 0; j < new_width; j++)
            padded_matrix[i][j] = 0;

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            padded_matrix[i + (filter_size / 2)][j + (filter_size / 2)] = matrix[i][j];
}

void apply_sobel(int** image, int x_size, int y_size, int kernel_x[3][3], int kernel_y[3][3], int kernel_size, int** output_image_gx, int** output_image_gy, int** output_image_magn, int** output_image_threshold, int threshold)
{
    // Gradient X
    int gx_min = 0;
    int gx_max = 255;
    for (int index_i = kernel_size / 2; index_i < y_size - (kernel_size / 2); ++index_i)
    {
        for (int index_j = kernel_size / 2; index_j < x_size - (kernel_size / 2); ++index_j)
        {
            float sum = 0;

            for (int i = -kernel_size / 2; i <= kernel_size / 2; ++i)
            {
                for (int j = -kernel_size / 2; j <= kernel_size / 2; ++j)
                {
                    float data = image[index_i + i][index_j + j];
                    float coeff = kernel_x[i + (kernel_size / 2)][j + (kernel_size / 2)];

                    sum += data * coeff;
                    
                    if (sum < gx_min)
                        gx_min = sum;
                    if (sum > gx_max)
                        gx_max = sum;


                }
            }
            output_image_gx[index_i - kernel_size / 2][index_j - kernel_size / 2] = sum;
            
        }
    }
    for (int index_i = kernel_size / 2; index_i < y_size - (kernel_size / 2); ++index_i)
    {
        for (int index_j = kernel_size / 2; index_j < x_size - (kernel_size / 2); ++index_j)
        {
            int value = output_image_gx[index_i - kernel_size / 2][index_j - kernel_size / 2];
            output_image_gx[index_i - kernel_size / 2][index_j - kernel_size / 2] =  255 * (value - gx_min) / (gx_max - gx_min);

        }
    }

    // Gradient Y
    int gy_min = 0;
    int gy_max = 255;
    for (int index_i = kernel_size / 2; index_i < y_size - (kernel_size / 2); ++index_i)
    {
        for (int index_j = kernel_size / 2; index_j < x_size - (kernel_size / 2); ++index_j)
        {
            float sum = 0;

            for (int i = -kernel_size / 2; i <= kernel_size / 2; ++i)
            {
                for (int j = -kernel_size / 2; j <= kernel_size / 2; ++j)
                {
                    float data = image[index_i + i][index_j + j];
                    float coeff = kernel_y[i + (kernel_size / 2)][j + (kernel_size / 2)];

                    sum += data * coeff;
                    
                    if (sum < gy_min)
                        gy_min = sum;
                    if (sum > gy_max)
                        gy_max = sum;
                }
            }
            output_image_gy[index_i - kernel_size / 2][index_j - kernel_size / 2] = sum;

        }
    }
    for (int index_i = kernel_size / 2; index_i < y_size - (kernel_size / 2); ++index_i)
    {
        for (int index_j = kernel_size / 2; index_j < x_size - (kernel_size / 2); ++index_j)
        {
            int value = output_image_gy[index_i - kernel_size / 2][index_j - kernel_size / 2];
            output_image_gy[index_i - kernel_size / 2][index_j - kernel_size / 2] = 255 * (value - gy_min) / (gy_max - gy_min);

        }
    }
    

    // Create magnitude image
    int magn_min = 0;
    int magn_max = 255;

    for (int i = 0; i < x_size; i++) {
        for (int j = 0; j < y_size; j++) {
            int xx = pow(output_image_gx[i][j], 2);
            int yy = pow(output_image_gy[i][j], 2);
            int magn = sqrt(xx + yy);

            if (magn < magn_min)
                magn_min = magn;
            if (magn > magn_max)
                magn_max = magn;

            output_image_magn[i][j] = magn;
        }
    }
    for (int index_i = 0; index_i < y_size; ++index_i)
    {
        for (int index_j = 0; index_j < x_size; ++index_j)
        {
            int value = output_image_magn[index_i][index_j];
            output_image_magn[index_i][index_j] = 255 * (value - magn_min) / (magn_max - magn_min);

        }
    }

    // Threshold image
    for (int i = 0; i < x_size; i++) {
        for (int j = 0; j < y_size; j++) {
            if (output_image_magn[i][j] > threshold)
                output_image_threshold[i][j] = output_image_magn[i][j];
        }
    }
}

int main()
{
    int sobel_X[3][3] =
    {
        {-1,0,1},
        {-2,0,2},
        {-1,0,1}
    };
    int sobel_y[3][3] =
    {
        {-1,-2,-1},
        { 0, 0, 0},
        { 1, 2, 1}
    };

    // lenna.pgm
    int **lenna_input, **input_padded, **lenna_gx, **lenna_gy, **lenna_mag, **lenna;
    int x_size, y_size, Q;
    char name[20] = "lenna.pgm";
    char outfile_gx[20] = "lenna_gx.pgm";
    char outfile_gy[20] = "lenna_gy.pgm";
    char outfile_mag[20] = "lenna_mag.pgm";
    char outfile[20] = "lenna_final.pgm";
    const int mask_size = 3;

    ReadImage(name, &lenna_input, x_size, y_size, Q);

    // Original Image
    cout << "Original Image" << endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            cout << lenna_input[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;

    // Pad image with zeros
    input_padded = new int* [y_size + mask_size - 1];
    for (int i = 0; i < y_size + mask_size - 1; i++)
        input_padded[i] = new int[x_size + mask_size - 1];
    padd_with_zeros_2D(lenna_input, input_padded, x_size, y_size, mask_size);

    cout << "Padded Image" << endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            cout << input_padded[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;

    // Perform sobel filtering
    lenna_gx = new int* [y_size];
    lenna_gy = new int* [y_size];
    lenna_mag = new int* [y_size];
    lenna = new int* [y_size];

    for (int i = 0; i < y_size; i++) {
        lenna_gx[i] = new int[x_size];
        lenna_gy[i] = new int[x_size];
        lenna_mag[i] = new int[x_size];
        lenna[i] = new int[x_size];
    }
    
    for (int i = 0; i < y_size; i++) {
        for (int j = 0; j < x_size; j++) {
            lenna_gx[i][j] = 0.0;
            lenna_gy[i][j] = 0.0;
            lenna_mag[i][j] = 0.0;
            lenna[i][j] = 0.0;
        }
    }

    apply_sobel(input_padded, x_size, y_size, sobel_X, sobel_y, 3, lenna_gx, lenna_gy, lenna_mag, lenna, 100);
    
    WriteImage(outfile_gx, lenna_gx, x_size, y_size, Q);
    WriteImage(outfile_gy, lenna_gy, x_size, y_size, Q);
    WriteImage(outfile_mag, lenna_mag, x_size, y_size, Q);
    WriteImage(outfile, lenna, x_size, y_size, Q);


    // Image after Sobel X
    cout << "Sobel X Image" << endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            cout << lenna_gx[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;

    // Image after Sobel Y
    cout << "Sobel Y Image" << endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            cout << lenna_gy[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;

    // Magnitude Image
    cout << "Magnitude Image" << endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            cout << lenna_mag[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;

    // Final Image
    cout << "Final Image" << endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            cout << lenna[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;


    waitKey(0);
    return 0;
}