package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	augmentedLagrangianMu        = 1.0
	augmentedLagrangianMuScale   = 10.0
	augmentedLagrangianMuMax     = 1e6
	augmentedLagrangianMaxOuter  = 20
	augmentedLagrangianMaxInner  = 50
	augmentedLagrangianTolerance = 1e-6
	augmentedLagrangianInnerStep = 0.01
)

// AugmentedLagrangian minimizuje konfigurisanu benchmark funkciju koristeći metod proširenog Lagranžijana: unutrašnja
// petlja gradijentnog spusta minimizuje funkciju uz penal i množioce, a spoljašnja petlja ažurira množioce i povećava
// penal koeficijent mu
// problem.Point je početna tačka pretrage
func AugmentedLagrangian(problem models.Problem) (result models.Result, err error) {

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

	mu := augmentedLagrangianMu
	if v, ok := problem.Payload["mu"].(float64); ok {
		mu = v
	}

	muScale := augmentedLagrangianMuScale
	if v, ok := problem.Payload["mu_scale"].(float64); ok {
		muScale = v
	}

	muMax := augmentedLagrangianMuMax
	if v, ok := problem.Payload["mu_max"].(float64); ok {
		muMax = v
	}

	maxOuter := augmentedLagrangianMaxOuter
	if v, ok := problem.Payload["max_outer"].(float64); ok {
		maxOuter = int(v)
	}

	maxInner := augmentedLagrangianMaxInner
	if v, ok := problem.Payload["max_inner"].(float64); ok {
		maxInner = int(v)
	}

	tolerance := augmentedLagrangianTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	lambdaAl := make([]float64, len(point))
	steps := 0

outer:
	for o := 0; o < maxOuter; o++ {

		innerLr := math.Min(augmentedLagrangianInnerStep, 1.0/(1.0+mu))

		for inner := 0; inner < maxInner; inner++ {
			gradient := fn.Gradient(point)
			for i := range point {
				gAug := gradient[i] + lambdaAl[i] + mu*point[i]
				point[i] -= innerLr * gAug
			}
			steps++
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}

		for i := range lambdaAl {
			lambdaAl[i] += mu * point[i]
		}
		mu = math.Min(mu*muScale, muMax)

		if norm(point) < tolerance {
			break outer
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "augmented_lagrangian",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
