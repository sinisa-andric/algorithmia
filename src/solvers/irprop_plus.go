package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	irpropPlusEtaPlus     = 1.2
	irpropPlusEtaMinus    = 0.5
	irpropPlusStepMin     = 1e-6
	irpropPlusStepMax     = 50.0
	irpropPlusMaxSteps    = 1000
	irpropPlusTolerance   = 1e-6
	irpropPlusInitialStep = 0.1
)

// IrpropPlus minimizuje konfigurisanu benchmark funkciju koristeći poboljšani Rprop sa backtracking-om (iRprop+): kor
// ak po dimenziji raste ili opada na osnovu znaka uzastopnih gradijenata,
// a pri promeni znaka se poslednji korak poništava ako je pogoršao rezultat
// problem.Point je početna tačka pretrage
func IrpropPlus(problem models.Problem) (result models.Result, err error) {

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

	etaPlus := irpropPlusEtaPlus
	if v, ok := problem.Payload["eta_plus"].(float64); ok {
		etaPlus = v
	}

	etaMinus := irpropPlusEtaMinus
	if v, ok := problem.Payload["eta_minus"].(float64); ok {
		etaMinus = v
	}

	stepMin := irpropPlusStepMin
	if v, ok := problem.Payload["step_min"].(float64); ok {
		stepMin = v
	}

	stepMax := irpropPlusStepMax
	if v, ok := problem.Payload["step_max"].(float64); ok {
		stepMax = v
	}

	maxSteps := irpropPlusMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := irpropPlusTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	point := append([]float64(nil), problem.Point...)
	delta := make([]float64, len(point))
	gradPrev := make([]float64, len(point))
	for i := range delta {
		delta[i] = irpropPlusInitialStep
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	pointPrev := append([]float64(nil), point...)
	valuePrev := fn.Evaluate(point)
	steps := 0

	for ; steps < maxSteps; steps++ {
		gradient := fn.Gradient(point)
		if norm(gradient) < tolerance {
			break
		}

		value := fn.Evaluate(point)

		for i := range point {
			product := gradient[i] * gradPrev[i]
			switch {
			case product > 0:
				delta[i] = min(delta[i]*etaPlus, stepMax)
				point[i] -= sign(gradient[i]) * delta[i]
				gradPrev[i] = gradient[i]
			case product < 0:
				delta[i] = max(delta[i]*etaMinus, stepMin)
				if value > valuePrev {
					point[i] = pointPrev[i]
				}
				gradPrev[i] = 0
			default:
				point[i] -= sign(gradient[i]) * delta[i]
				gradPrev[i] = gradient[i]
			}
		}

		pointPrev = append([]float64(nil), point...)
		valuePrev = value

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "irprop_plus",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
