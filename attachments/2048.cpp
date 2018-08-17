#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdlib>
#define SIZE 4

using namespace std;

struct ArrayDigits{
	int x;
	int y;
};

void PrintArray (int m[SIZE][SIZE]);
void Move (int (&m)[SIZE][SIZE], string Direction, bool &can_move);
void RandomGenerator (int (&m)[SIZE][SIZE]);
void game_over_checker(int m[SIZE][SIZE], bool &no_moves);

int main(){

	ifstream infile; 
	string filename;
	int x;

	int m[SIZE][SIZE];

	cout << "enter initial configuration file name: " << endl;
	cin >> filename;

	infile.open(filename.c_str());

	if(!infile.is_open()){

		cout << "file not found, using default start configuration: " << endl;
				int i = 0;
				int j = 0;
				while (i < (SIZE - 1)){
					while (j < (SIZE - 1)){
						m[i][j] = 0;
						j++;
					}											//assigns all the elements of array, m, the value of 0 
				i++;
				}
				m[3][3] = 2;									//except for the last element
				PrintArray(m);
	}
	
	else{
			for (int i = 0; i < SIZE; i++){
				for (int j = 0; j <SIZE; j++){
					infile >> m[i][j];							//reads from the input file stream
				}
			}
			PrintArray(m);	
		}
		cout << endl;
		infile.close();

		string drct;
		bool no_moves = false;

		while (no_moves == false){
			cin >> drct;										//input direction of motion 
			cout << endl;
			bool can_move = false;
			Move(m, drct, can_move);							//function called that makes the elements move (and adds)
			if (can_move == true){
				RandomGenerator(m);								//random number generated in place of a zero
				PrintArray(m);
				cout << endl;
			}
			else{
				game_over_checker(m, no_moves);					//checks if any possible move left
			}
		}
		cout << "game over" << endl;							//if no possible moves, exits while loop 

	return 0;
}

void PrintArray (int m[SIZE][SIZE]){							//sole purpose to print array
	for (int i = 0; i < SIZE; i++){								//i represents rows (throughout the program)
					for (int j = 0; j < SIZE; j++){				//j represents columns (throughout the program)
						cout << m[i][j] << "\t";		
					}
					cout << endl;								//shifts to next line after 4 elements printed
	}
}

void Move (int (&m)[SIZE][SIZE], string Direction, bool &can_move){
	if (Direction == "a"){ 										//for direction LEFT
		for (int i = 0; i < SIZE; i++){	
			for (int n = 0; n < SIZE; n++){						//n represents number of times the while loop runs on column j again				
				for (int j = 0; j < SIZE -1; j++){			
					if((m[i][j] == 0 && m[i][j+1] != 0)){		//checks if value of element is zero and value of element to the right is non-zero
						can_move = true; 
						for(int k = j; k < SIZE - 1; k++){		
							m[i][k] = m[i][k+1];
						}
						m[i][3] = 0;
					}
				}
			}
			for (int j = 0; j < SIZE -1; j++){
				if ((m[i][j] == m[i][j+1] && m[i][j]!= 0)){			//checks if value of element is equal to value of element on the right AND none of the elements has value zero
					can_move = true;
					m[i][j] = 2*m[i][j];
					for(int k = j+1; k < SIZE -1; k++){
						m[i][k] = m[i][k+1];
					}
					m[i][3] = 0;
				}

			}
		}

	}

	else if(Direction == "d"){ 										//for direction RIGHT
		for (int i = 0; i < SIZE; i++){				
			for (int n = 0; n < SIZE; n++){			
				for (int j = SIZE - 1; j >0; j--){
					if((m[i][j] == 0) && (m[i][j-1] != 0)){			//checks if value of element is zero and value of element to the left is non-zero
						can_move = true;
						for(int k = j; k > 0; k--){
							m[i][k] = m[i][k-1];
						}
						m[i][0] = 0;
					}
				}
			}
			for (int j = SIZE -1; j >= 0; j--){
				if ((m[i][j] == m[i][j-1]) && (m[i][j] != 0)){			//checks if value of element is equal to value of element on the left AND none of the elements has value zero
					can_move = true;
					m[i][j] = 2*m[i][j];
					for(int k = j - 1; k > 0; k--){
						m[i][k] = m[i][k-1];
					}
					m[i][0] = 0;
				}
			}
		}
	}

	else if(Direction == "w"){ 											//for direction UP
		for (int j = 0; j < SIZE; j++){
			for (int n = 0; n < SIZE; n++){
				for (int i = 0; i < SIZE - 1; i++){
					if((m[i][j] == 0) && (m[i+1][j] != 0)){ 			//checks if value of element is zero and value of element one step down is non-zero
						can_move = true;
						for(int k = i; k < SIZE - 1; k++){
							m[k][j] = m[k+1][j];
						}
						m[3][j] = 0;
					}
				}
			}
			for (int i = 0; i < SIZE -1; i++){
				if ((m[i][j] == m[i +1][j]) && (m[i][j] != 0)){			//checks if value of element is equal to value of element one step down AND none of the elements has value zero
					can_move = true;
					m[i][j] = 2*m[i][j];
					for(int k = i+1; k < SIZE -1; k++){
						m[k][j] = m[k+1][j];
					}
				m[3][j] = 0;
				}

			}
		}
	}

	else if (Direction == "s"){											//for direction DOWN
		for (int j = 0; j < SIZE; j++){
			for (int n = 0; n < SIZE; n++){
				for (int i = SIZE - 1; i >0; i--){
					if((m[i][j] == 0) && (m[i-1][j] != 0)){				//checks if value of element is zero and value of element one step up is non-zero
						can_move = true;
						for(int k = i; k > 0; k--){
							m[k][j] = m[k-1][j];
						}
						m[0][j] = 0;
					}
				}
			}
			for (int i = SIZE -1; i >= 0; i--){
				if ((m[i][j] == m[i -1][j]) && (m[i][j] != 0)){			//checks if value of element is equal to value of element one step up AND none of the elements has value zero
					can_move = true;
					m[i][j] = 2*m[i][j];
					for(int k = i - 1; k > 0; k--){
						m[k][j] = m[k-1][j];
					}
					m[0][j] = 0;
				}

			}
		}
	}
}

void RandomGenerator (int (&m)[SIZE][SIZE]){
		int k = 0;														//k represents the index of "elements of the compound"
		ArrayDigits a[SIZE*SIZE];										//ArrayDigits is the compound structure
		for (int j = 0; j <SIZE; j++){
			for (int i = 0; i < SIZE; i++){
				if(m[i][j] == 0){
					a[k].x = i;											//if value of element is zero
					a[k].y = j;											//"elements of the compound" are assigned values which represents the position of the element of the array
					k++;												
				}
			}
		}
		int b = rand() % k;												//random integer between 0 and (k-1) chosen
		m[a[b].x][a[b].y] = 2;											//random "elements of the compound" are passed as position of element of the array
}																		//value of 2 assigned

void game_over_checker(int m[SIZE][SIZE], bool &no_moves){				
	bool move_checker = false;											//initially, no moves possible
	for (int i = 0; i < SIZE && (move_checker!= true); i++){
		for (int j = 0; j < SIZE && (move_checker!= true); j++){
			if (m[i][j] == 0){											//if the value of any element is zero
				move_checker = true;									//move is possible
			}
		}
	}

	for (int i = 0; i < SIZE && (move_checker !=true); i++){
		for (int j = 0; j < (SIZE -1) && (move_checker != true); j++){
			if (m[i][j] == m[i][j+1]){									//if the value of an element and the value of the element to the right is equal
				move_checker = true;									//move is possible
			}
		}
	}

	for (int j = 0; j < SIZE && (move_checker !=true); j++){
		for (int i = 0; i < (SIZE -1) && (move_checker != true); i++){
			if (m[i][j] == m[i+1][j]){									//if the value of an element and the value of the element one step down is equal
				move_checker = true;									//move is possible
			}
		}
	}

	if (move_checker == false){											//if no moves possible as yet
		no_moves = true;												//no possible moves can be made
	}
}