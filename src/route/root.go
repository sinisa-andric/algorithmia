package route

import (
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type ProblemParams struct {
	Data string `json:"data,omitempty"`
}

type AlgorithmiaResponse struct {
	Approved bool        `json:"approved,omitempty"`
	Message  string      `json:"message,omitempty"`
	Response interface{} `json:"response,omitempty"`
}

// Handler function
func SolveProblem(ctx *gin.Context) {

	log := zap.New(nil)

	defer log.Sync() // Flush logs before exiting

	requestId := ctx.GetString("id")
	log = log.Named("[Algorithmia:ProblemHandler]").WithOptions(
		zap.Fields(
			zap.String("requestId", requestId),
		),
	)

	log.Info("problem started")

	var problemParams ProblemParams

	if err := ctx.BindJSON(&problemParams); err != nil {
		log.Error("failed to bind request body", zap.Error(err))
		ctx.JSON(http.StatusBadRequest,
			AlgorithmiaResponse{
				Approved: false,
				Message:  "failed to bind request body",
				Response: nil,
			},
		)
		return
	}

	response, err := solveProblem(problemParams.Data)
	if err != nil {
		log.Error("problem solving failed", zap.Error(err))
		ctx.JSON(http.StatusInternalServerError,
			AlgorithmiaResponse{
				Approved: false,
				Message:  "not solved",
				Response: nil,
			},
		)
		return
	}

	log.Info("solve problem finished")

	ctx.JSON(http.StatusCreated,
		AlgorithmiaResponse{
			Approved: true,
			Message:  "solved!",
			Response: response,
		},
	)

}

func solveProblem(data string) (result interface{}, err error) {

	if data == "" {
		return nil, fmt.Errorf("input data is not provided")
	}

	return nil, nil
}
