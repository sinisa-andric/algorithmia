package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	svrgLearningRate     = 0.1
	svrgInnerSteps       = 10
	svrgSnapshotInterval = 20
	svrgMaxSteps         = 1000
	svrgTolerance        = 1e-6
	svrgStepClip         = 1.0
)

// Svrg minimizuje konfigurisanu benchmark funkciju koristeći Stochastic Variance Reduced Gradient: svakih
// snapshot_interval koraka se pamti referentni gradijent u trenutnoj tački (snapshot), a korak koristi korekciju
// gradijenta u odnosu na taj snapshot ublaženu pokretnim prosekom gradijenata od poslednjeg snapshot-a (prozor
// veličine inner_steps) — eksplicitna redukcija varijanse preko čuvane referentne tačke, za razliku od svih
// ostalih EMA/momentum pristupa u ovom modulu
// problem.Point je početna tačka pretrage
func Svrg(problem models.Problem) (result models.Result, err error) {

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

	learningRate := svrgLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	innerSteps := svrgInnerSteps
	if v, ok := problem.Payload["inner_steps"].(float64); ok {
		innerSteps = int(v)
	}
	if innerSteps < 1 {
		innerSteps = 1
	}

	snapshotInterval := svrgSnapshotInterval
	if v, ok := problem.Payload["snapshot_interval"].(float64); ok {
		snapshotInterval = int(v)
	}
	if snapshotInterval < 1 {
		snapshotInterval = 1
	}

	maxSteps := svrgMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := svrgTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x := append([]float64(nil), problem.Point...)
	snapshotGrad := make([]float64, dimensions)
	window := make([][]float64, 0, innerSteps)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		if steps%snapshotInterval == 0 {
			snapshotGrad = numGrad(fn.Evaluate, x, proximalGradEps)
			window = window[:0]
		}

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		if len(window) < innerSteps {
			window = append(window, grad)
		} else {
			window[steps%innerSteps] = grad
		}

		avgSinceSnapshot := make([]float64, dimensions)
		for _, g := range window {
			for d := range avgSinceSnapshot {
				avgSinceSnapshot[d] += g[d]
			}
		}
		for d := range avgSinceSnapshot {
			avgSinceSnapshot[d] /= float64(len(window))
		}

		delta := make([]float64, dimensions)
		for d := range delta {
			correction := grad[d] - snapshotGrad[d] + avgSinceSnapshot[d]
			delta[d] = -learningRate * correction
		}
		delta = clipStep(delta, svrgStepClip)
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
		Method:     "svrg",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
