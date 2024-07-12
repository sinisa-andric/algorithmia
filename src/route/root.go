package route

import (
	"fmt"
	"math/rand/v2"
	"net/http"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type ProblemParams struct {
	Data  string    `json:"data,omitempty"`
	Point []float64 `json:"point,omitempty"`
}

type AlgorithmiaResponse struct {
	Approved bool        `json:"approved"`
	Message  string      `json:"message,omitempty"`
	Response interface{} `json:"response,omitempty"`
}

// Handler function
func SolveProblemHandler(ctx *gin.Context) {

	log := zap.New(nil)

	defer log.Sync() // Flush logs before exiting

	requestId := ctx.GetString("id")
	log = log.Named("[Algorithmia:ProblemHandler]").WithOptions(
		zap.Fields(
			zap.String("requestId", requestId),
		),
	)

	log.Info("solve problem started")

	var problemParams ProblemParams

	err := ctx.BindJSON(&problemParams)
	if err != nil {
		log.Error("failed to bind request body", zap.Error(err))
		ctx.JSON(
			http.StatusBadRequest,
			AlgorithmiaResponse{
				Approved: false,
				Message:  "failed to bind request body: " + err.Error(),
				Response: nil,
			},
		)
		return
	}

	response, err := SolveProblem(problemParams.Data, problemParams.Point)
	if err != nil {
		ctx.JSON(
			http.StatusNotAcceptable,
			AlgorithmiaResponse{
				Approved: false,
				Message:  "not solved: " + err.Error(),
				Response: response,
			},
		)
		return
	}

	ctx.JSON(
		http.StatusCreated,
		AlgorithmiaResponse{
			Approved: true,
			Message:  "solved!",
			Response: response,
		},
	)

	log.Info("solve problem finished",
		zap.Any("points", problemParams.Point),
		zap.Any("data", problemParams.Data),
	)

}

func SolveProblem(data string, point []float64) (result interface{}, err error) {

	if data == "" {
		err = fmt.Errorf("input data is not provided")
		return nil, err
	}

	// logic
	randomArray := RandomArray(len(point))

	result = ProblemParams{
		Data:  "return random value",
		Point: randomArray,
	}

	return result, nil
}

func RandomArray(arrayLength int) (result []float64) {

	result = make([]float64, arrayLength)
	for i := range result {
		result[i] = rand.Float64()
	}

	return result
}
