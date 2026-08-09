package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	majorizationMinimizationMaxSteps     = 1000
	majorizationMinimizationLearningRate = 0.1
	majorizationMinimizationLambda       = 0.01
	majorizationMinimizationLInit        = 1.0
	majorizationMinimizationTolerance    = 1e-6
	majorizationMinimizationStepClip     = 1.0
	majorizationMinimizationMaxBacktrack = 10
)

// MajorizationMinimization minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći Majorization-
// Minimization: eksplicitna backtracking pretraga Lipšicove konstante L pre svakog koraka dok kvadratni majorant
// ne nadmaši stvarnu vrednost funkcije, za razliku od forward_backward_ls.go koji radi backtracking direktno na
// vrednosti kompozitne ciljne funkcije
// problem.Point je početna tačka pretrage
func MajorizationMinimization(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := majorizationMinimizationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	lambda := majorizationMinimizationLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	lInit := majorizationMinimizationLInit
	if v, ok := problem.Payload["L_init"].(float64); ok {
		lInit = v
	}

	tolerance := majorizationMinimizationTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	x := append([]float64(nil), problem.Point...)
	l := lInit

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		fx := fn.Evaluate(x)
		candidate := x
		for range majorizationMinimizationMaxBacktrack {
			raw := make([]float64, dimensions)
			for d := range raw {
				raw[d] = x[d] - (1/l)*grad[d]
			}
			candidate = proxL1(raw, lambda/l)

			dot := 0.0
			sqDist := 0.0
			for d := range candidate {
				diff := candidate[d] - x[d]
				dot += grad[d] * diff
				sqDist += diff * diff
			}
			majorant := fx + dot + (l/2)*sqDist
			if fn.Evaluate(candidate) <= majorant {
				break
			}
			l *= 2
		}

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = candidate[d] - x[d]
		}
		delta = clipStep(delta, majorizationMinimizationStepClip)
		for d := range x {
			x[d] += delta[d]
		}

		l = math.Max(l*0.9, lInit)

		value := objective(x)
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
		Method:     "majorization_minimization",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
