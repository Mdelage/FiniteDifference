"""Scripts to find the finite difference discretization for any derivative at any order"""
import numpy as np


def _format_number_str(number):
    form = 'st' if number == 1 else 'nd' if number == 2 else 'rd' if number == 3 else 'th'
    return str(number) + form


class FiniteDifference:
    def __init__(self, derivative, order, list_of_points):
        """
        :param int derivative: Which derivative to find a scheme for
        :param int order: The order of precision of the scheme
        :param list_of_points: The list of index of points to use for the scheme (e.g., [3, 2, 1, 0, -1, -2])
        """
        self.derivative = derivative
        self.order = order
        self.list_of_points = list_of_points
        self.size_of_scheme = self.order + self.derivative

        self.coeff_matrix = None
        self.inverse_matrix = None
        self.list_of_coeffs = None

        self.check_order()
        self.check_points()

    def check_order(self):
        if self.order < self.derivative:
            raise ValueError(f"Order of precision wanted strictly less than"
                             f" the degree of derivative! ({self.order} < {self.derivative}")

    def check_points(self):
        if self.size_of_scheme != len(self.list_of_points):
            raise ValueError(f"Incorrect number of points ({len(self.list_of_points)})"
                             f" for order and derivative chosen "
                             f"({self.derivative} + {self.order} = {self.order + self.derivative})!")

    def build_coeff_matrix(self):
        size = self.size_of_scheme
        coeff_matrix = np.empty((size, size))

        for j in range(size):
            for i in range(size):
                coeff_matrix[i, j] = self.list_of_points[j] ** i / np.math.factorial(i)

        self.coeff_matrix = coeff_matrix

    def build_scheme(self):
        self.build_coeff_matrix()

        try:
            inverse_matrix = np.linalg.inv(self.coeff_matrix)
        except np.linalg.LinAlgError as e:
            print("Coefficient matrix non-inversible!")
            raise e

        self.inverse_matrix = inverse_matrix

        vec = np.zeros(self.size_of_scheme)
        vec[self.derivative] = 1
        self.list_of_coeffs = np.matmul(inverse_matrix, vec)

    def print_scheme(self, extended=False):
        if self.list_of_coeffs is None:
            print("Scheme not yet computed!")
        else:
            der_str = f"d{self.derivative}udx{self.derivative}_i"
            u_str = [f"u_i{index:+}" for index in self.list_of_points]
            scheme_str = ''
            for j in range(self.size_of_scheme):
                scheme_str += str(self.list_of_coeffs[j]) + u_str[j]
                if j < self.size_of_scheme - 1:
                    scheme_str += " + "
            order_str = _format_number_str(self.order)
            derivative_str = _format_number_str(self.derivative)

            print(order_str + "-order finite difference approximation for " + derivative_str + " derivative")
            print(der_str + " = " + f"1/DeltaX^{self.derivative} * (" + scheme_str + f") + O(DeltaX^{self.order})")

            if extended:
                print()
                print("Coefficient matrix:")
                print(self.coeff_matrix)
                print()
                print("Inverse coefficient matrix")
                print(self.inverse_matrix)


if __name__ == '__main__':
    scheme = FiniteDifference(2, 4, [3, 2, 1, 0, -1, -2])
    scheme.build_scheme()
    scheme.print_scheme(extended=True)
