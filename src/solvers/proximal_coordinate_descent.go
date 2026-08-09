package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	proximalCoordinateDescentMaxSteps     = 1000
	proximalCoordinateDescentLearningRate = 0.1
	proximalCoordinateDescentLambda       = 0.01
	proximalCoordinateDescentTolerance    = 1e-6
	proximalCoordinateDescentStepClip     = 1.0
)

// ProximalCoordinateDescent minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći proksimalni
// koordinatni spust: jedna nasumično izabrana dimenzija po iteraciji dobija gradijent+soft-threshold korak, za
// razliku od postojećeg coordinate_descent.go koji nema L1/prox deo, samo čist gradijentni korak po koordinati
// problem.Point je početna tačka pretrage
func ProximalCoordinateDescent(problem models.Problem) (result models.Result, err error) {

	if len(problem.Point) == 0 {
		err = fmt.Errorf("starting point is required")
		return result, err
	}

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}
	if err := fn.ValidateDimension(len(problem.Point)); err != nil {
		return result, err
	}

	maxSteps := proximalCoordinateDescentMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := proximalCoordinateDescentLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := proximalCoordinateDescentLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	tolerance := proximalCoordinateDescentTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	x := append([]float64(nil), problem.Point...)

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		d := rand.IntN(dimensions)
		plus := append([]float64(nil), x...)
		minus := append([]float64(nil), x...)
		plus[d] += proximalGradEps
		minus[d] -= proximalGradEps
		partialGrad := (fn.Evaluate(plus) - fn.Evaluate(minus)) / (2 * proximalGradEps)

		candidate := proxL1Scalar(x[d]-learningRate*partialGrad, learningRate*lambda)
		x[d] += clamp(candidate-x[d], -proximalCoordinateDescentStepClip, proximalCoordinateDescentStepClip)

		value := objective(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}

		if includeTrajectory {
			// bestValue prati objective (fn + L1 penal), dok rezultat vraća čist fn.Evaluate(best) — koristi se
			// isti izraz kao Value: ispod da bi putanja bila konzistentna sa finalnim rezultatom
			trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), false)
		}

		if math.Abs(bestValue-prevBestValue) < tolerance {
			noImprove++
		} else {
			noImprove = 0
		}
		if noImprove >= maxNoImprove {
			steps++
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "proximal_coordinate_descent",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
