/*
    Author: Mohamed Ashraf (Wx)
    Date: 2/3/2022
    Type: General Purpose Library for Neural Network.
*/

// HEADER GUARDS: __Wx_NeuralNetwork_C_H__
#ifndef __NN_H1_H__
#define __NN_H1_H__

// HEADER LIBRARIES USED:
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <conio.h>
#include <unistd.h>
#include <windows.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

//=====================================

// IMPORTANT KEYBOARD KEYS(ASCII):
// BACKSPACE:
#define BKSP 8
// ENTER:
#define ENTER 13
// ESCAPE:
#define ESC 27
// SPACE:
#define SPACE 32
// TAB:
#define TAB 9

//==================================================

/*****************************************************
*                    GLOBAL VARIABLES                *
******************************************************/

/*
Function input format
[INPUT_LAYER_DENSE, (HIDDEN_LAYER_NUMBER, HIDDEN_LAYER_DENSE), OUTPUT_LAYER_DENSE]

*/

// GENERAL PURPOSE
#define INPUT_LAYER_DENSE    2
#define OUTPUT_LAYER_DENSE   1
#define HIDDEN_LAYERS_DENSE  10

#define HIDDEN_LAYER_NUMBER  2

// BACKPROPAGATION
#define LEARNING_RATE        0.01

/*****************************************************
*                  NEURAL NETWORK STRUCTS            *
******************************************************/

// Input Layer Neuron struct control.
typedef struct
{
  double NEURON[INPUT_LAYER_DENSE];

}INPUT_NEURONS;

// Hidden Layers Neuron struct control.
typedef struct
{
  //uint16_t LAYERS_N;

  double NEURON[HIDDEN_LAYERS_DENSE];

}HIDDEN_NEURONS;


// Output Layer Neuron struct control.
typedef struct
{
  double NEURON[INPUT_LAYER_DENSE];

}OUTPUT_NEURONS;

// Struct to store the Layers Weights
typedef struct
{
  double **WEIGHTS;

}WEIGHTS;

// Struct to store the Layers Biases
typedef struct
{
  double BIASES;

}BIASES;

// Global Access Structs.
INPUT_NEURONS  i_l;
HIDDEN_NEURONS h_l[HIDDEN_LAYER_NUMBER];
OUTPUT_NEURONS o_l;

// Weights number max = HIDDEN_LAYERS_NUMBER + 1 (1 for the output layer).
WEIGHTS weights[ (HIDDEN_LAYER_NUMBER + 1) ];

// Biases number max = HIDDEN_LAYERS_NUMBER + 1 (1 for the output layer).
BIASES biases[ (HIDDEN_LAYER_NUMBER + 1) ];

/*****************************************************
*                    FUNCTION DECLARED               *
******************************************************/
double MEAN_SQUARE_ERROR(double *);


/*****************************************************
*                 NEURAL NETWORK FUNCTIONS           *
******************************************************/





/*                # BACKPROPGATION FUNCTIONS #         */

// Function to calculate the Mean Square Error (MSE)
/*
- Mean Square Error Forumla:
    E = 0.5 * [SUM|( 0->(N-1) )| ( (Actual - Desired) pow 2 )].

  ** "SUPERVISED LEARNING".
*/
double MEAN_SQUARE_ERROR(double *_ACTUAL_OUTPUT)
{
  // Variable to store the Mean Square Error.
  double _MSE = 0.0;

  // Variable Array to store the desired output values.
  double _DESIRED_OUTPUT[OUTPUT_LAYER_DENSE] = {0.0};

  for(uint16_t i = 0; (i < OUTPUT_LAYER_DENSE); i++)
  {
    _MSE += pow((_DESIRED_OUTPUT[i] - _ACTUAL_OUTPUT[i]), 2);
  }

  _MSE = ( 0.5 * _MSE );

  return _MSE;
}

/*****************************************************
*                    MATH FUNCTIONS                  *
******************************************************/

// Function to apply the Matrix Vector Multiplication.
void VxM(const double *VECTOR,
         const double **MATRIX,
         double *RESULT,
         const uint16_t row,
         const uint16_t col)
{

  double sum = 0.0;

  for(uint16_t i = 0; (i < row); i++)
  {
    for(uint16_t j = 0; (j < col); j++)
    {
      sum += VECTOR[i] * MATRIX[i][j];
    }
    RESULT[i] = sum;
    sum = 0;
  }

}

/*****************************************************
*                    SUB FUNCTIONS                   *
******************************************************/


#endif
