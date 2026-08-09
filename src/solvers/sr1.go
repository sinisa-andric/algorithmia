package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	sr1MaxSteps  = 1000
	sr1Tolerance = 1e-6
	sr1StepClip  = 1.0
)

// Sr1 minimizuje konfigurisanu benchmark funkciju koristeći SR1 (Symmetric Rank-One) kvazi-Newton metod: H se
// ažurira RANG-JEDAN formulom (outer(r,r)/dot(r,y), r=s-H*y), za razliku od DFP/BFGS koji su rang-2 update-i —
// ažuriranje se PRESKAČE kad je imenilac dot(r,y) preblizu nuli u odnosu na ||r||*||y|| (poznata SR1 numerička
// nestabilnost), umesto da se primeni nestabilna deoba
// problem.Point je početna tačka pretrage
func Sr1(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := sr1MaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := sr1Tolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	x := append([]float64(nil), problem.Point...)
	h := identityMatrix(dimensions)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		p := negate(matVec(h, grad))

		alpha := backtrackAlpha(fn.Evaluate, x, p)
		s := make([]float64, dimensions)
		for d := range s {
			s[d] = alpha * p[d]
		}
		s = clipStep(s, sr1StepClip)

		xNew := make([]float64, dimensions)
		for d := range xNew {
			xNew[d] = x[d] + s[d]
		}

		gradNew := numGrad(fn.Evaluate, xNew, proximalGradEps)
		y := make([]float64, dimensions)
		for d := range y {
			y[d] = gradNew[d] - grad[d]
		}

		hy := matVec(h, y)
		r := make([]float64, dimensions)
		for d := range r {
			r[d] = s[d] - hy[d]
		}

		ry := dot(r, y)
		if math.Abs(ry) > 1e-8*norm(r)*norm(y) {
			rr := outer(r, r)
			for i := range h {
				for j := range h[i] {
					h[i][j] += rr[i][j] / ry
				}
			}
		}

		x = xNew

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
		Method:     "sr1",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
