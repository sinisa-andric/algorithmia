package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	dynamicRelaxationMass      = 1.0
	dynamicRelaxationDamping   = 0.8
	dynamicRelaxationDT        = 0.01
	dynamicRelaxationMaxSteps  = 10000
	dynamicRelaxationTolerance = 1e-6
)

// DynamicRelaxation minimizuje konfigurisanu benchmark funkciju koristeći dinamičku relaksaciju: simulira prigušenu
// česticu date mase koja se kreće u polju sile force = -gradient(x) dok se ne umiri
// problem.Point je početna pozicija čestice
func DynamicRelaxation(problem models.Problem) (result models.Result, err error) {

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

	mass := dynamicRelaxationMass
	if v, ok := problem.Payload["mass"].(float64); ok {
		mass = v
	}

	damping := dynamicRelaxationDamping
	if v, ok := problem.Payload["damping"].(float64); ok {
		damping = v
	}

	dt := dynamicRelaxationDT
	if v, ok := problem.Payload["dt"].(float64); ok {
		dt = v
	}

	maxSteps := dynamicRelaxationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := dynamicRelaxationTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	velocity := make([]float64, len(point))

	best := append([]float64(nil), point...)
	bestValue := fn.Evaluate(point)

	steps := 0
	for ; steps < maxSteps; steps++ {

		gradient := fn.Gradient(point)

		for i := range point {
			force := -gradient[i]
			acceleration := force / mass
			velocity[i] = damping*velocity[i] + dt*acceleration
			point[i] += dt * velocity[i]
		}

		if value := fn.Evaluate(point); value < bestValue {
			bestValue = value
			best = append([]float64(nil), point...)
		}
		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}

		if norm(velocity) < tolerance {
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "dynamic_relaxation",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
