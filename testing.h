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

//==================================================

/*****************************************************
*                    GLOBAL VARIABLES                *
******************************************************/

/*
Function input format
[INPUT_LAYER_DENSE, (HIDDEN_LAYER_NUMBER, HIDDEN_LAYER_DENSE), OUTPUT_LAYER_DENSE]

*/

// GENERAL PURPOSE: <NETWORK TOPOLOGY>
#define INPUT_LAYER_DENSE    ((const uint16_t) 2U)
#define OUTPUT_LAYER_DENSE   ((const uint16_t) 1U)
#define HIDDEN_LAYERS_DENSE  ((const uint16_t) 3U)

#define HIDDEN_LAYER_NUMBER  ((const uint16_t) 1U)

#define BETA                 ((const double) 0.01)
#define SLOPE                ((const double) 0.02)
// BACKPROPAGATION
#define LEARNING_RATE        ((const double) 0.01U)


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
  double WEIGHT[10][10];

}WEIGHTS;

// Struct to store the Layers Biases
typedef struct
{
  double BIAS;

}BIASES;

// Global Access Structs.
INPUT_NEURONS  i_l;
HIDDEN_NEURONS h_l[HIDDEN_LAYER_NUMBER];
OUTPUT_NEURONS o_l;

// Weights number max = HIDDEN_LAYERS_NUMBER + 1 (1 for the output layer).
#define WEIGHTS_NUMBERS ((const uint16_t) (HIDDEN_LAYER_NUMBER + 1))
WEIGHTS weights[WEIGHTS_NUMBERS];

// Biases number max = HIDDEN_LAYERS_NUMBER + 1 (1 for the output layer).
#define BIASES_NUMBERS  ((const uint16_t) (HIDDEN_LAYER_NUMBER + 1))
BIASES biases[BIASES_NUMBERS];

/*****************************************************
*                    FUNCTION DECLARED               *
******************************************************/
double *VxM(const double *,
            const double (*)[10],
            double *,
            const uint16_t,
            const uint16_t);

double MEAN_SQUARE_ERROR(double *);
void forwardpropagation_algorithm(void);
void Apply_ActivationFunction(double *, const uint16_t, const char);
void ADD_BIAS(double *, const uint16_t);


/*****************************************************
*                 NEURAL NETWORK FUNCTIONS           *
******************************************************/

/*              # FORWARD PROPGATION FUNCTIONS #             */
/*
  Weight Matrix Row = Number Of Previous Layer Neurons Dense.
  Weight Matrix Col = Number Of Current  Layer Neurons Dense.

  [X0] [W0,0 W1,0 Wn,0]     [Y0]
  [X1] [W0,1 W1,1 Wn,1] ==> [Y1] ==> g(Y[n++]) ==> Next_Layer
  [Xn] [W0,2 W2,2 Wn,j]     [Yn]

  i: Next_Neuron_Length.
  j: Prev_Neuron_Length.

  We can do convertion of metrices ROWxCOL using [Matrix_Transposition] Theory.
*/

/*
 * Asumed All Data are initalized.
   - Algorithm Outlines:
    1- Input x Weights(INPUT->HIDDEN[0]).
      - Apply The Activation Function.
    2- Loop till Hidden[n] x Weights[HIDDEN[0]->HIDDEN[N-1]].
      - Apply The Activation Function.
    3- Hidden[N-1] * Wights(Hidden[N-1]->OUTPUT).
      - Apply The Activation Function.
*/
void forwardpropagation_algorithm(void)
{
  uint16_t counter = 0;

  // First: Input x Weights(INPUT->HIDDEN[0]).
  VxM(i_l.NEURON, weights[0].WEIGHT, h_l[0].NEURON, HIDDEN_LAYERS_DENSE,
                                                    INPUT_LAYER_DENSE);
  //Apply The Activiation Function: sigmoid()
  Apply_ActivationFunction(h_l[0].NEURON, HIDDEN_LAYERS_DENSE, 's');

  // Second: Hidden[n] x Weights(HIDDEN[n]->HIDDEN[n+1]).
  // If there is multi hidden layers.
  if(HIDDEN_LAYER_NUMBER > 0)
  {
    while( (counter < (HIDDEN_LAYER_NUMBER)))
    {
      VxM(h_l[counter].NEURON, weights[counter+1].WEIGHT,
          h_l[counter+1].NEURON,
          HIDDEN_LAYERS_DENSE,
          HIDDEN_LAYERS_DENSE);

      // ADD Bias.
      ADD_BIAS(h_l[counter].NEURON, counter);

      // Apply The Activation Function: sigmoid()
      Apply_ActivationFunction(h_l[counter].NEURON, HIDDEN_LAYERS_DENSE, 's');
      counter += 1;
    }
  }

  // Third: Hidden[n-1] x Weights(HIDDEN[n-1]->OUTPUT).
  VxM(h_l[HIDDEN_LAYER_NUMBER-1].NEURON,
      weights[WEIGHTS_NUMBERS-1].WEIGHT,
      o_l.NEURON,
      OUTPUT_LAYER_DENSE,
      HIDDEN_LAYERS_DENSE);

  // ADD Bias.
  ADD_BIAS(o_l.NEURON, BIASES_NUMBERS);

  // Apply The Activation Function: sigmoid()
  Apply_ActivationFunction(o_l.NEURON, OUTPUT_LAYER_DENSE, 's');

}//end forwardpropagation_algorithm.


/*                # BACKPROPGATION FUNCTIONS #             */
/*
- Mean Square Error Forumla:
    E = 0.5 * [SUM|( 0->(N-1) )| ( (Actual - Desired) pow 2 )].

  ** "SUPERVISED LEARNING".
*/
// Function to Calculate the output mean square error.
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
}// end MEAN_SQUARE_ERROR.


/*
  *- Back Propagation Math:

  =Normal Multi Variable Calculus:

  MSE = 0.5(Desired - Actual) pow 2.
  MSE = 0.5(Desired - Sigmoid(O_N)) pow2. :: O_N => Current Output Neuron Weights Sum.
  dE/dW = d(MSE)/dW :: Which is Complicated to be calculated.

  =By sChain Rule:

  | [Input] <- [Weights] <- [Weights_Sum] <- [Prediction_Output] <- [Prediction_Error] s|

  ********************************************
  *  MSE: Mean Square Error.                 *
  *  Wn: Weight [0,n].                       *
  *  P_O: Predicted Output.                  *
  *  D_O: Desired Output.                    *
  *  W_S: Weighted SUM (SUM OF WEIGHTS).     *
  ********************************************

  d(E)/d(Wn) = [ d(MSE)/d(P_O) * d(P_O)/d(W_S) * d(W_S)/d(Wn) ]

  >Now we can diffrentiate each term indvidual to simplify the calculations.

  dMSE/d(P_O) = d( [ 0.5 * (D_O - P_O)pow 2 ] )/d(P_O) : Diffrentitate with respect to P_O.
  dMSE/d(P_O) = P_O - D_O

  d(P_O)/d(W_S) = d(Sigmoid(W_S))/d(W_S) : Diffrentitate with respect to W_S.
  <NOTE> d(Sigmoid(x))/dx = Sigmoid(x)(1-Sigmoid(x)) <NOTE>
  d(P_O)/d(W_S) = Sigmoid(x)(1-Sigmoid(x)

  d(W_S)/d(Wn) = d(SUM(Wn * Xn) + Bns)/d(Wn) : Differentiate with respect to Wn <'N's Could be 1, 2 .. n>.

  >After substitution now we have [d(MSE)/d(Wn)].

  [ Wn_NEW = Wn_OLD - (Learnin_Rate) * d(MSE)/d(Wn) ] : Weights Error Update Formula.
*/
// Function to implement Back Propagation Algorithm.
void backpropagation_algorithm(void)
{

}

/*****************************************************
*              ACTIVATION FUNCTIONS                  *
******************************************************/

// Linear Activation Function.
double linear_af(double INPUT)
{
  return INPUT;
}//end linear_af.

// Linear Activation Function.
double relu_af(double INPUT)
{
  if(INPUT <= 0)
  {
    return 0;
  }
  else
  {
    return INPUT;
  }
}//end relu_af.

// Drevative of Relu Activation Function.
double drelu_af(double INPUT)
{
  if(INPUT < 0)
  {
    return 0;
  }
  else if(INPUT >= 0)
  {
    return 1;
  }

  else
    return -1;
}//end drelu_af.

// Sigmoid Activation Function.
double sigmoid_af(double INPUT)
{
  return ( 1/(1-exp( (-1) * INPUT) ) );
}//end sigmoid.

// Drevative Of Sigmoid Activation Function.
double dsigmoid_af(double INPUT)
{
  return ( sigmoid_af(INPUT)*(1 - sigmoid_af(INPUT)) );
}//end dsigmoid_af.

// Tanh Activation Function.
double tanh_af(double INPUT)
{
  return ( tanh(INPUT) );
}//end tanh_af.

// Drevative Of Tanh Activation Function.
double dtanh_af(double INPUT)
{
  return ( 1 - pow(tanh(INPUT), 2) );
}//end dtanh_af.

// Swish Activation Function.
double swish_af(double INPUT)
{
  return (INPUT * sigmoid_af(INPUT * BETA));
}//end switch_af.

// Drevative Of Swish Activation Function.
double dswish_af(double INPUT)
{
  return ( (BETA * swish_af(INPUT)) + (sigmoid_af(INPUT) * (1-(BETA*swish_af(INPUT)))) );
}//end dswish_af.

// Leaky Relu (eRELU) Activation Function
double lrelu_af(double INPUT)
{
  if(INPUT > 0)
    return INPUT;

  else if(INPUT <= 0)
    return (SLOPE * INPUT);

  else
    return -1; // Error Detected.
}//end lrelu_af.

// Drevative Leaky Relu (ELU) Activation Function
double dlrelu_af(double INPUT)
{
  //const double CONSTANT = 1;

  if(INPUT > 0)
    return 1;

  else if(INPUT <= 0)
    return SLOPE;

  else
    return -1; // Error Detected.
}//end dlrelu_af.

/*****************************************************
*                    MATH FUNCTIONS                  *
******************************************************/

// Function to apply the Matrix Vector Multiplication.
/*
  #VECTOR: The Input Vector (Previous Layer).
  #MATRIX: The Weight Matrix Between Previous And Next Layer.
  #RESULT: The Output Vector (Next Layer).
  #ROW:    The Output  Vector length - The Weights Matrix ROW.
  #COL:    The Input   Vector Length - The Weights COL.
*/
double *VxM(const double *VECTOR,
            const double (*MATRIX)[10],
            double *RESULT,
            const uint16_t row,
            const uint16_t col)
{
  double sum = 0.0;

  // Getting The Transpose of weights matrix.

  // Swapping the indecis.
  /*
    Apply the transpose only if VECTOR[X][0], MATRIX[M][N] -- (X != N).
  */


  // Calculating the Dot Product.
  for(uint16_t i = 0; (i < row); i++)
  {
    for(uint16_t j = 0; (j < col); j++)
    {
      sum += VECTOR[j] * MATRIX[i][j];
    }
    RESULT[i] = sum;
    sum = 0.0;
  }

  return RESULT;
}

/*****************************************************
*                    SUB FUNCTIONS                   *
******************************************************/

// Function to initalize the Neural Network Data
/*
  - Initalizing ALgorithm:

  [(INPUTS)]->[(HIDDEN_0)] .         . [(HIDDEN_N)]->[(OUTPUT)]
            |              |         |              |
      Weights[INPUT]  Weights[0]  Weights[n]  Weights[OUTPUT]
                      Bias[0]     Bias[n]     Bias[OUTPUT]
*/
void NN_INIT(void)
{
  // First Initalizing the input layer. (testing purposes)
  for(uint16_t i = 0; (i < INPUT_LAYER_DENSE); i++)
    i_l.NEURON[i] = (rand()%200) / 10.0;


  // Initalizing Weights[INPUT].

  for(uint32_t i = 0; (i < HIDDEN_LAYERS_DENSE); i++)
  {
    for(uint32_t j = 0; (j < INPUT_LAYER_DENSE); j++)
    {
      weights[0].WEIGHT[i][j] = (rand()%250) / 32.0;
    }//end for(j).
  }//end for(i).

  // Initalizing Weights[HIDDEN] - [0,inf)
  for(uint32_t i = 0; (i < HIDDEN_LAYER_NUMBER); i++)
  {
    for(uint32_t j = 0; (j < HIDDEN_LAYERS_DENSE); j++)
    {
      for(uint32_t k = 0; (k < HIDDEN_LAYERS_DENSE); k++)
      {
        weights[i].WEIGHT[j][k] = (rand()%250) / 50.0;
      }//end for(k).
    }//end for(j).
  }//end for(i).

  // Initalizing Weights[OUTPUT]
  for(uint32_t i = 0; (i < OUTPUT_LAYER_DENSE); i++)
  {
    for(uint32_t j = 0; (j < HIDDEN_LAYERS_DENSE); j++)
    {
      weights[ (WEIGHTS_NUMBERS - 1) ].WEIGHT[i][j] = (rand()%250) / 66.0;
    }//end for(j).
  }//end for(i).

  // Initalizing Biases[0:(BIASES_NUMBERS - 1)]
  for(uint32_t i = 0; (i < OUTPUT_LAYER_DENSE); i++)
  {
    for(uint32_t j = 0; (j < HIDDEN_LAYERS_DENSE); j++)
    {
      biases[ (WEIGHTS_NUMBERS - 1) ].BIAS = (rand()%250) / 2342.0;
    }//end for(j).
  }//end for(i).

}//end NN_INIT.


//Function to print Vector.
void PRINT_VECTOR(double *VECTOR, const uint16_t SIZE)
{
  for(uint16_t i = 0; (i < SIZE); i++)
  {
    printf("\n%lf", VECTOR[i]);
  }
}//end PRINT_VECTOR.

// Function to print Matrix.
void PRINT_MATRIX(double (*MATRIX)[10], const uint16_t ROW,
                                   const uint16_t COL)
{
  for(uint16_t i = 0; (i < ROW); i++)
  {
    for(uint16_t j = 0; (j < COL); j++)
    {
      printf("%lf\t", MATRIX[i][j]);
    }
    printf("\n");
  }
}//end PRINT_MATRIX.

// Function to Add The bias to the hidden_layer_vector.
void ADD_BIAS(double *VECTOR, const uint16_t current_layer)
{
  for(uint16_t i = 0; (i < HIDDEN_LAYER_NUMBER); i++)
  {
    VECTOR[i] = VECTOR[i] + biases[current_layer].BIAS;
  }
}//end ADD_BIAS.

// Easy picking the activation function.
void Apply_ActivationFunction(double *VECTOR, const uint16_t SIZE, const char ACTIVATION_FUNCTION)
{
  switch(ACTIVATION_FUNCTION)
  {
    // Linear.
    case 'l':
    {
      for(uint16_t i = 0; (i < SIZE); i++)
        VECTOR[i] = linear_af(VECTOR[i]);
      break;
    }

    // Relu.
    case 'r':
    {
      for(uint16_t i = 0; (i < SIZE); i++)
        VECTOR[i] = relu_af(VECTOR[i]);
      break;
    }

    // Sigmoid.
    case 's':
    {
      for(uint16_t i = 0; (i < SIZE); i++)
        VECTOR[i] = sigmoid_af(VECTOR[i]);
      break;
    }
    // Drevative Sigmoid.
    case '1':
    {
      for(uint16_t i = 0; (i < SIZE); i++)
        VECTOR[i] = dsigmoid_af(VECTOR[i]);
      break;
    }
    // Tanh.
    case 't':
    {
      for(uint16_t i = 0; (i < SIZE); i++)
        VECTOR[i] = tanh_af(VECTOR[i]);
      break;
    }
    // Drevative Tanh.
    case '2':
    {
      for(uint16_t i = 0; (i < SIZE); i++)
        VECTOR[i] = dtanh_af(VECTOR[i]);
      break;
    }

    default: {printf("\n Error /-Activation_Functions-/\n"); break;}
  }
}//end Apply_ActivationFunction.

#endif /* _NN_C_H_ */
