package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	shampooLearningRate = 0.1
	shampooEpsilon      = 1e-4
	shampooMaxSteps     = 1000
	shampooTolerance    = 1e-6
	shampooStepClip     = 1.0
)

// Shampoo minimizuje konfigurisanu benchmark funkciju koristeći POJEDNOSTAVLJENU dijagonalnu aproksimaciju
// Shampoo optimizatora: prva dimenzija akumulira sopstveni "levi" statistički akumulator L, a sve preostale
// dimenzije dele zajednički "desni" akumulator R — NAPOMENA: pravi Shampoo faktorizuje punu matricu težina preko
// Kronecker-ovih proizvoda i koristi matrične korene, što je van dometa ovog sistema bez linalg biblioteke; ovo je
// dijagonalna aproksimacija te ideje
// problem.Point je početna tačka pretrage
func Shampoo(problem models.Problem) (result models.Result, err error) {

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

	learningRate := shampooLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	epsilon := shampooEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := shampooMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := shampooTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	l, r := 0.0, 0.0

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		l += grad[0] * grad[0]

		delta := make([]float64, dimensions)
		delta[0] = -learningRate * grad[0] / math.Sqrt(l+epsilon)

		if dimensions > 1 {
			restSumSq := 0.0
			for d := 1; d < dimensions; d++ {
				restSumSq += grad[d] * grad[d]
			}
			r += restSumSq / float64(dimensions-1)
			for d := 1; d < dimensions; d++ {
				delta[d] = -learningRate * grad[d] / math.Sqrt(r+epsilon)
			}
		}

		delta = clipStep(delta, shampooStepClip)
		for d := range x {
			x[d] += delta[d]
		}

		value := fn.Evaluate(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}
		if includeTrajectory {
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
		Method:     "shampoo",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
