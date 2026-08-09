package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	grokfastLearningRate = 0.1
	grokfastAlphaGrok    = 0.98
	grokfastLambGrok     = 2.0
	grokfastMaxSteps     = 1000
	grokfastTolerance    = 1e-6
	grokfastStepClip     = 1.0
)

// Grokfast minimizuje konfigurisanu benchmark funkciju koristeći Grokfast: sporo opadajući EMA gradijenta se
// POJAČAVA i dodaje nazad na trenutni gradijent (ne zamenjuje ga kao momentum.go), pojačavajući niskofrekventnu
// komponentu signala gradijenta
// problem.Point je početna tačka pretrage
func Grokfast(problem models.Problem) (result models.Result, err error) {

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

	learningRate := grokfastLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	alphaGrok := grokfastAlphaGrok
	if v, ok := problem.Payload["alpha_grok"].(float64); ok {
		alphaGrok = v
	}

	lambGrok := grokfastLambGrok
	if v, ok := problem.Payload["lamb_grok"].(float64); ok {
		lambGrok = v
	}

	maxSteps := grokfastMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := grokfastTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	slowGrad := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		delta := make([]float64, dimensions)
		for d := range delta {
			slowGrad[d] = alphaGrok*slowGrad[d] + (1-alphaGrok)*grad[d]
			amplified := grad[d] + lambGrok*slowGrad[d]
			delta[d] = -learningRate * amplified
		}
		delta = clipStep(delta, grokfastStepClip)
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
		Method:     "grokfast",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
