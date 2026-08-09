package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	lbfgsLearningRate = 0.1
	lbfgsHistorySize  = 10
	lbfgsMaxSteps     = 1000
	lbfgsTolerance    = 1e-6
)

// Lbfgs minimizuje konfigurisanu benchmark funkciju koristeći Limited-Memory BFGS: inverzni Hesijan se aproksimira
// preko dvostruke rekurzije nad klizećim prozorom nedavnih razlika pozicije i gradijenta
// problem.Point je početna tačka pretrage
func Lbfgs(problem models.Problem) (result models.Result, err error) {

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

	learningRate := lbfgsLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	historySize := lbfgsHistorySize
	if v, ok := problem.Payload["history_size"].(float64); ok {
		historySize = int(v)
	}

	maxSteps := lbfgsMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := lbfgsTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	gradient := fn.Gradient(point)

	var historyS, historyY [][]float64
	steps := 0

	for ; steps < maxSteps; steps++ {
		if norm(gradient) < tolerance {
			break
		}

		n := len(historyS)
		q := append([]float64(nil), gradient...)
		alphas := make([]float64, n)
		rhos := make([]float64, n)

		for i := n - 1; i >= 0; i-- {
			rhos[i] = 1 / dot(historyY[i], historyS[i])
			alphas[i] = rhos[i] * dot(historyS[i], q)
			for j := range q {
				q[j] -= alphas[i] * historyY[i][j]
			}
		}

		r := q
		for i := 0; i < n; i++ {
			beta := rhos[i] * dot(historyY[i], r)
			for j := range r {
				r[j] += historyS[i][j] * (alphas[i] - beta)
			}
		}

		direction := negate(r)

		newPoint := make([]float64, len(point))
		for i := range point {
			newPoint[i] = point[i] + learningRate*direction[i]
		}
		newGradient := fn.Gradient(newPoint)

		s := make([]float64, len(point))
		y := make([]float64, len(point))
		for i := range point {
			s[i] = newPoint[i] - point[i]
			y[i] = newGradient[i] - gradient[i]
		}

		if dot(y, s) > 1e-10 {
			historyS = append(historyS, s)
			historyY = append(historyY, y)
			if len(historyS) > historySize {
				historyS = historyS[1:]
				historyY = historyY[1:]
			}
		}

		point = newPoint
		gradient = newGradient

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "lbfgs",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
