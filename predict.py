from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from itertools import combinations
from joblib import dump, load
import numpy as np
from collections import defaultdict

suits = ['c', 'd', 'h', 's']  # clubs, diamonds, hearts, spades
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T',
         'J', 'Q', 'K', 'A']  # 2-10, Jack, Queen, King, Ace

deck = {rank + suit: i for i, (rank, suit) in enumerate((r, s)
                                                        for r in ranks for s in suits)}
card_order_dict = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
                   "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


def check_straight_flush(hand):
    if check_flush(hand) and check_straight(hand):
        return True
    else:
        return False


def check_four_of_a_kind(hand):
    values = [i[0] for i in hand]
    value_counts = defaultdict(lambda: 0)
    for v in values:
        value_counts[v] += 1
    if sorted(value_counts.values()) == [1, 4]:
        return True
    return False


def check_full_house(hand):
    values = [i[0] for i in hand]
    value_counts = defaultdict(lambda: 0)
    for v in values:
        value_counts[v] += 1
    if sorted(value_counts.values()) == [2, 3]:
        return True
    return False


def check_flush(hand):
    suits = [i[1] for i in hand]
    if len(set(suits)) == 1:
        return True
    else:
        return False


def check_straight(hand):
    values = [i[0] for i in hand]
    value_counts = defaultdict(lambda: 0)
    for v in values:
        value_counts[v] += 1
    rank_values = [card_order_dict[i] for i in values]
    value_range = max(rank_values) - min(rank_values)
    if len(set(value_counts.values())) == 1 and (value_range == 4):
        return True
    else:
        # check straight with low Ace
        if set(values) == set(["A", "2", "3", "4", "5"]):
            return True
        return False


def check_three_of_a_kind(hand):
    values = [i[0] for i in hand]
    value_counts = defaultdict(lambda: 0)
    for v in values:
        value_counts[v] += 1
    if set(value_counts.values()) == set([3, 1]):
        return True
    else:
        return False


def check_two_pairs(hand):
    values = [i[0] for i in hand]
    value_counts = defaultdict(lambda: 0)
    for v in values:
        value_counts[v] += 1
    if sorted(value_counts.values()) == [1, 2, 2]:
        return True
    else:
        return False


def check_one_pairs(hand):
    values = [i[0] for i in hand]
    value_counts = defaultdict(lambda: 0)
    for v in values:
        value_counts[v] += 1
    if 2 in value_counts.values():
        return True
    else:
        return False


mlb = MultiLabelBinarizer()

all_combinations = []
for r in range(2, 6):
    all_combinations.extend(combinations(deck.values(), r))

mlb.fit(all_combinations)
loaded_model = load('./model/model.sav')
unhinted_loaded_model = load('./model/unhinted_model.sav')

le = LabelEncoder()
le.fit(['Flush', 'Four of a Kind', 'Full House', 'High Card', 'One Pair',
       'Royal Flush', 'Straight', 'Straight Flush', 'Three of a Kind', 'Two Pairs'])


def check_hand(hand):
    combin = list(combinations(hand, 5))
    if len(hand) < 5:
        combin = [hand]
    highest_comb = 0
    for comb in combin:
        if check_straight_flush(comb):
            highest_comb = 9
        elif check_four_of_a_kind(comb):
            if highest_comb < 8:
                highest_comb = 8
        elif check_full_house(comb):
            if highest_comb < 7:
                highest_comb = 7
        elif check_flush(comb):
            if highest_comb < 6:
                highest_comb = 6
        elif check_straight(comb):
            if highest_comb < 5:
                highest_comb = 5
        elif check_three_of_a_kind(comb):
            if highest_comb < 4:
                highest_comb = 4
        elif check_two_pairs(comb):
            if highest_comb < 3:
                highest_comb = 3
        elif check_one_pairs(comb):
            if highest_comb < 2:
                highest_comb = 2
        else:
            if highest_comb < 1:
                highest_comb = 1

    comb = {1: "High Card", 2: "One Pair", 3: "Two Pairs", 4: "Three of a Kind",
            5: "Straight", 6: "Flush", 7: "Full House", 8: "Four of a Kind", 9: "Straight Flush"}
    return comb[highest_comb]


def predict(hand):
    highest_combo = check_hand(hand)
    hand = [deck[card] for card in hand]

    hand_combo = le.transform([highest_combo]).reshape(1, -1)

    hand = mlb.transform([hand])
    return loaded_model.predict(np.concatenate((hand, hand_combo), axis=1))[0]


def unhinted_predict(hand):
    hand = [deck[card] for card in hand]
    hand = mlb.transform([hand])
    return unhinted_loaded_model.predict(hand)[0]


if __name__ == '__main__':
    import random

    # fold_count = {0: 0, 1: 0}
    # keys = list(deck.keys())

    # for _ in range(1000):
    #     hand = random.sample(keys, 5)
    #     fold_count[unhinted_predict(hand)] += 1
    # print(fold_count)
    # print('fold_percent: ' + str(fold_count[1] / sum(fold_count.values())))

    # new_data = ['7d', 'Ad', 'Ah', 'As', 'Qh', 'Qs', 'Th']
    # print(predict(new_data))

    keys = list(deck.keys())
    new_data = random.sample(keys, 5)
    print(new_data)
    print(predict(new_data))
    print(unhinted_predict(new_data))
