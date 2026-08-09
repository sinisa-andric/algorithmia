package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	adafactorLearningRate = 0.1
	adafactorEpsilon1     = 1e-30
	adafactorEpsilon2     = 1e-3
	adafactorBeta2        = 0.999
	adafactorMaxSteps     = 1000
	adafactorTolerance    = 1e-6
	adafactorStepClip     = 1.0
)

// Adafactor minimizuje konfigurisanu benchmark funkciju koristeći pojednostavljenu dijagonalnu aproksimaciju
// Adafactor optimizatora: umesto pune EMA drugog momenta po dimenziji (kao Adam), drugi moment se FAKTORIZUJE u dva
// skalara — red (prva polovina dimenzija) i kolona (druga polovina) — čiji se proizvod deljen prosekom koristi kao
// zajednička skala za sve dimenzije. NAPOMENA: ovo je pojednostavljena verzija — pravi Adafactor faktorizuje punu
// matricu težina preko Kronecker-ovog proizvoda, što je van dometa ovog sistema bez linalg biblioteke
// problem.Point je početna tačka pretrage
func Adafactor(problem models.Problem) (result models.Result, err error) {

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

	learningRate := adafactorLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	epsilon1 := adafactorEpsilon1
	if v, ok := problem.Payload["epsilon1"].(float64); ok {
		epsilon1 = v
	}

	epsilon2 := adafactorEpsilon2
	if v, ok := problem.Payload["epsilon2"].(float64); ok {
		epsilon2 = v
	}

	beta2 := adafactorBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	maxSteps := adafactorMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := adafactorTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	half := dimensions / 2
	if half < 1 {
		half = 1
	}

	x := append([]float64(nil), problem.Point...)
	r, c := 0.0, 0.0

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)

		rowMeanSq := 0.0
		for d := 0; d < half; d++ {
			rowMeanSq += grad[d] * grad[d]
		}
		rowMeanSq /= float64(half)

		colMeanSq := 0.0
		colCount := dimensions - half
		if colCount < 1 {
			colCount = 1
		}
		for d := half; d < dimensions; d++ {
			colMeanSq += grad[d] * grad[d]
		}
		colMeanSq /= float64(colCount)

		r = beta2*r + (1-beta2)*rowMeanSq
		c = beta2*c + (1-beta2)*colMeanSq

		sApprox := r * c / ((r+c)/2 + epsilon1)

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = -learningRate * grad[d] / (math.Sqrt(sApprox) + epsilon2)
		}
		delta = clipStep(delta, adafactorStepClip)
		for d := range x {
			x[d] += delta[d]
		}

		value := fn.Evaluate(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
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
		Method:     "adafactor",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
