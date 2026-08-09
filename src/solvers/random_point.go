package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"math/rand/v2"
)

// RandomPoint minimizuje konfigurisanu benchmark funkciju koristeći nasumično uzorkovanje: povlači jednu nasumičnu
// tačku uniformno iz [0, 1] po dimenziji, bez ikakve iterativne pretrage
// problem.Point samo određuje dimenzionalnost rezultata
func RandomPoint(problem models.Problem) (result models.Result, err error) {

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}
	if err := fn.ValidateDimension(len(problem.Point)); err != nil {
		return result, err
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := make([]float64, len(problem.Point))
	for i := range point {
		point[i] = rand.Float64()
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, 1, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "random_point",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      1,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
