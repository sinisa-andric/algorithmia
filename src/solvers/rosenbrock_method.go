package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	rosenbrockMethodMaxSteps  = 1000
	rosenbrockMethodTolerance = 1e-6
	rosenbrockMethodStepSize  = 0.1
)

// RosenbrockMethod minimizuje konfigurisanu benchmark funkciju koristeći Rosenbrock-ov metod rotirajućih koordinata:
// pretražuje duž skupa smerova koji se Gram-Schmidt-om ponovo ortogonalizuju kad god svaki smer uspe u jednom prolazu
// problem.Point je početna tačka pretrage
func RosenbrockMethod(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := rosenbrockMethodMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := rosenbrockMethodTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	stepSize := rosenbrockMethodStepSize
	if v, ok := problem.Payload["step_size"].(float64); ok {
		stepSize = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	n := len(point)

	directions := make([][]float64, n)
	for i := range directions {
		directions[i] = make([]float64, n)
		directions[i][i] = 1
	}

	alpha := make([]float64, n)
	for i := range alpha {
		alpha[i] = stepSize
	}
	delta := make([]float64, n)
	success := make([]bool, n)

	steps := 0
	for ; steps < maxSteps; steps++ {

		value := fn.Evaluate(point)

		for i := 0; i < n; i++ {
			newPoint := make([]float64, n)
			for j := range point {
				newPoint[j] = point[j] + alpha[i]*directions[i][j]
			}
			newValue := fn.Evaluate(newPoint)

			if newValue < value {
				point = newPoint
				value = newValue
				delta[i] += alpha[i]
				alpha[i] *= 1.5
				success[i] = true
			} else {
				alpha[i] *= -0.5
			}
		}

		allSuccess := true
		for _, s := range success {
			if !s {
				allSuccess = false
				break
			}
		}

		if allSuccess {
			directions = gramSchmidtRotate(directions, delta)
			for i := range alpha {
				alpha[i] = stepSize
				delta[i] = 0
				success[i] = false
			}
		}

		maxAlpha := 0.0
		for _, a := range alpha {
			if math.Abs(a) > maxAlpha {
				maxAlpha = math.Abs(a)
			}
		}
		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}

		if maxAlpha < tolerance {
			steps++
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "rosenbrock_method",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}

// gramSchmidtRotate gradi novi ortonormirani skup smerova pretrage iz
// akumuliranog uspešnog pomeraja (delta) duž trenutnih smerova, prateći
// Rosenbrock-ovu šemu rotacije.
func gramSchmidtRotate(directions [][]float64, delta []float64) [][]float64 {

	n := len(directions)
	a := make([][]float64, n)
	for i := 0; i < n; i++ {
		a[i] = make([]float64, n)
		for j := i; j < n; j++ {
			for k := range a[i] {
				a[i][k] += delta[j] * directions[j][k]
			}
		}
	}

	newDirections := make([][]float64, n)
	for i := 0; i < n; i++ {
		b := append([]float64(nil), a[i]...)
		for k := 0; k < i; k++ {
			proj := dot(a[i], newDirections[k])
			for c := range b {
				b[c] -= proj * newDirections[k][c]
			}
		}

		bn := norm(b)
		if bn < 1e-10 {
			newDirections[i] = directions[i]
			continue
		}
		for c := range b {
			b[c] /= bn
		}
		newDirections[i] = b
	}

	return newDirections
}
