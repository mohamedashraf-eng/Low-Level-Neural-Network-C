#include "testing.h"


void print_matrix(double (*)[3], uint8_t, uint8_t );
double sigmoid(double );
double desired_output(double , double);


int main(void)
{
  system("COLOR 0A");

  double input[2] = {0};
  double output   = 0;

  double hidden[2][3] = {0};

  double weight_ih1[2][3];
  double weight_h1h2[3][3];
  double weight_h2o[3][1];

  double bias[3] = {0.1332, 0.2549, 0.2420};

  //init the input.
  srand(time(NULL));
  for(uint8_t i = 0; i < 2; i++)
    input[i] = (rand()%2) / 1.0;

  //init the weights.

  // input to hidden 1
  for(uint8_t i = 0; i < 2; i++)
    for(uint8_t j = 0; j < 3; j++)
      weight_ih1[i][j] = (rand()%50) / 150.0;

  // hidden 1 to hidden 2.
  for(uint8_t i = 0; i < 3; i++)
    for(uint8_t j = 0; j < 3; j++)
      weight_h1h2[i][j] = (rand()%50) / 150.0;

  // hidden 2 to output.
  for(uint8_t i = 0; i < 3; i++)
    for(uint8_t j = 0; j < 1; j++)
      weight_h2o[i][j] = (rand()%50) / 200.0;


  // forward propgation algorithm.
  // input to hidden 1
  hidden[0][0] = input[0]*weight_ih1[0][0] +
                 input[1]*weight_ih1[1][0];
  hidden[0][0] = sigmoid(hidden[0][0] + bias[0]);

  hidden[0][1] = input[0]*weight_ih1[0][1] +
                 input[1]*weight_ih1[1][1];
  hidden[0][1] = sigmoid(hidden[0][1] + bias[0]);

  hidden[0][2] = input[0]*weight_ih1[0][2] +
                 input[1]*weight_ih1[1][2];
  hidden[0][2] = sigmoid(hidden[0][2] + bias[0]);


  // hidden 1 to hidden 2
  hidden[1][0] = hidden[0][0]*weight_h1h2[0][0] +
                 hidden[0][1]*weight_h1h2[1][0] +
                 hidden[0][2]*weight_h1h2[2][0];
  hidden[1][0] = sigmoid(hidden[1][0] + bias[1]);

  hidden[1][1] = hidden[0][0]*weight_h1h2[0][1] +
                 hidden[0][1]*weight_h1h2[1][1] +
                 hidden[0][2]*weight_h1h2[2][1];
  hidden[1][1] = sigmoid(hidden[1][1] + bias[1]);

  hidden[1][2] = hidden[0][0]*weight_h1h2[0][2] +
                 hidden[0][1]*weight_h1h2[1][2] +
                 hidden[0][2]*weight_h1h2[2][2];
  hidden[1][2] = sigmoid(hidden[1][2] + bias[1]);

  // hidden 2 to output
  output = hidden[1][0]*weight_h2o[0][0] +
           hidden[1][1]*weight_h2o[1][0] +
           hidden[1][2]*weight_h2o[2][0];
  output = sigmoid(output + bias[2]);


  // Backpropgation algorithm.
  double mse = 0.0; // mean square error.

  mse = 0.5 * pow( (desired_output(input[0], input[1]) - output), 2);
  




 // printing status.
  printf("\n Input: \n");
  for(uint8_t i = 0; i < 2; i++)
    printf(" %.2f\n", input[i]);

  printf("\n W_(Input->Hidden1):\n");
  print_matrix(weight_ih1, 2, 3);

  printf("\n Hidden 1: \n");
  for(uint8_t i = 0; i < 3; i++)
    printf(" %.4f\n", hidden[0][i]);


  printf("\n W_(Hidden1->Hidden2):\n");
  print_matrix(weight_h1h2, 3, 3);

  printf("\n Hidden 2: \n");
  for(uint8_t i = 0; i < 3; i++)
    printf(" %.4f\n", hidden[1][i]);

  printf("\n W_(Hidden2->Output):\n");
  for(uint8_t i = 0; i < 3; i++)
  {
    for(uint8_t j = 0; j < 1; j++)
     {
        printf(" %.4f", weight_h2o[i][j]);
     }
     printf("\n");
  }
  printf("\n");


  printf("\n actual_output: %f \n", output);
  printf(" desired_output: %f \n", desired_output(input[0], input[1]));
  printf("\n mean square error: %f", mse);

  printf("\n");
  return 0;
}

void print_matrix(double (*m)[3], uint8_t a, uint8_t b)
{
  for(uint8_t i = 0; i < a; i++)
  {
    for(uint8_t j = 0; j < b; j++)
     {
        printf(" %.4f\t ", m[i][j]);
     }
     printf("\n");
  }
  printf("\n");
}

double sigmoid(double x)
{
  return ( 1/(1+exp((-1) * x)) );
}

double desired_output(double in1, double in2)
{
  return ( (uint8_t) in1 ^ (uint8_t) in2 );
}
