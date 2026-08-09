package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	adamoLearningRate = 0.01
	adamoBeta1        = 0.9
	adamoBeta2        = 0.999
	adamoEpsilon      = 1e-8
	adamoLambda       = 0.01
	adamoMaxSteps     = 1000
	adamoTolerance    = 1e-6
)

// AdamO minimizuje konfigurisanu benchmark funkciju koristeći Adam optimizer sa ortogonalnom regularizacijom: standar
// dni Adam korak se dopunjuje projekcijom parametara ka ortogonalnom prostoru (Stiefel manifold) kroz penalizaciju
// ||x^T x - I||², što sprečava kolaps parametara u jednu dimenziju
// problem.Point je početna tačka pretrage i određuje dimenzionalnost
func AdamO(problem models.Problem) (result models.Result, err error) {

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

	learningRate := adamoLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	beta1 := adamoBeta1
	if v, ok := problem.Payload["beta1"].(float64); ok {
		beta1 = v
	}

	beta2 := adamoBeta2
	if v, ok := problem.Payload["beta2"].(float64); ok {
		beta2 = v
	}

	epsilon := adamoEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	lambda := adamoLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	maxSteps := adamoMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := adamoTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	m := make([]float64, len(point))
	v := make([]float64, len(point))
	steps := 0

	for ; steps < maxSteps; steps++ {
		gradient := fn.Gradient(point)
		if norm(gradient) < tolerance {
			break
		}

		t := float64(steps + 1)
		beta1T := math.Pow(beta1, t)
		beta2T := math.Pow(beta2, t)

		for i := range point {
			m[i] = beta1*m[i] + (1-beta1)*gradient[i]
			v[i] = beta2*v[i] + (1-beta2)*gradient[i]*gradient[i]

			mHat := m[i] / (1 - beta1T)
			vHat := v[i] / (1 - beta2T)

			point[i] -= learningRate * mHat / (math.Sqrt(vHat) + epsilon)
		}

		// Ortogonalna regularizacija ima smisla samo kad postoji "rotacioni" stepen
		// slobode, tj. kad je ||x|| > 1 — inače vuče parametre od nule i sprečava
		// konvergenciju ka malim vrednostima blizu globalnog minimuma
		norm2 := dot(point, point)
		if norm2 > 1.0 {
			scale := lambda * learningRate / norm2
			for i := range point {
				point[i] -= scale * point[i] * norm2
			}
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "adamo",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
