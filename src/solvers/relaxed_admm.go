package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	relaxedAdmmMaxSteps     = 1000
	relaxedAdmmLearningRate = 0.1
	relaxedAdmmLambda       = 0.01
	relaxedAdmmRho          = 1.0
	relaxedAdmmAlphaRelax   = 1.6
	relaxedAdmmTolerance    = 1e-6
	relaxedAdmmStepClip     = 1.0
)

// RelaxedAdmm minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći ADMM sa over-relaxation: koristi
// relaksiranu kombinaciju x i z (faktor alpha_relax>1) u z/u ažuriranjima umesto direktnog x^{k+1}, za razliku od
// admm.go koji koristi x^{k+1} direktno bez relaksacije
// problem.Point je početna tačka pretrage
func RelaxedAdmm(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := relaxedAdmmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := relaxedAdmmLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := relaxedAdmmLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	rho := relaxedAdmmRho
	if v, ok := problem.Payload["rho"].(float64); ok {
		rho = v
	}

	alphaRelax := relaxedAdmmAlphaRelax
	if v, ok := problem.Payload["alpha_relax"].(float64); ok {
		alphaRelax = v
	}

	tolerance := relaxedAdmmTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	x := append([]float64(nil), problem.Point...)
	z := append([]float64(nil), problem.Point...)
	u := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = -learningRate * (grad[d] + rho*(x[d]-z[d]+u[d]))
		}
		delta = clipStep(delta, relaxedAdmmStepClip)
		for d := range x {
			x[d] += delta[d]
		}

		xRelaxed := make([]float64, dimensions)
		for d := range xRelaxed {
			xRelaxed[d] = alphaRelax*x[d] + (1-alphaRelax)*z[d]
		}

		zRaw := make([]float64, dimensions)
		for d := range zRaw {
			zRaw[d] = xRelaxed[d] + u[d]
		}
		z = proxL1(zRaw, lambda/(rho+1e-10))

		for d := range u {
			u[d] += xRelaxed[d] - z[d]
		}

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
		Method:     "relaxed_admm",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
