package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	yogiLearningRate = 0.1
	yogiBeta1        = 0.9
	yogiBeta2        = 0.999
	yogiEpsilon      = 1e-3
	yogiMaxSteps     = 1000
	yogiTolerance    = 1e-6
	yogiStepClip     = 1.0
)

// Yogi minimizuje konfigurisanu benchmark funkciju koristeći Yogi optimizator: drugi moment se ažurira ADITIVNOM
// korekcijom u pravcu predznaka razlike (s-grad^2) umesto eksponencijalnog opadanja kao Adam, čime se sprečava
// prebrzo opadanje efektivnog learning rate-a
// problem.Point je početna tačka pretrage
func Yogi(problem models.Problem) (result models.Result, err error) {

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

	learningRate := yogiLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := yogiBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := yogiBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := yogiEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := yogiMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := yogiTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	m := make([]float64, dimensions)
	s := make([]float64, dimensions)

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
			m[d] = beta1*m[d] + (1-beta1)*grad[d]
			g2 := grad[d] * grad[d]
			s[d] = s[d] - (1-beta2)*sign(s[d]-g2)*g2
			if s[d] < 0 {
				s[d] = 0
			}
			delta[d] = -learningRate * m[d] / (math.Sqrt(s[d]) + epsilon)
		}
		delta = clipStep(delta, yogiStepClip)
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
		Method:     "yogi",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
