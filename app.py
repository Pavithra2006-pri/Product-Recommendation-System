from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    abort,
    session,
    redirect,
    url_for
)

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from difflib import get_close_matches

import os


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = "shopsmart-cart-secret-key"


# ============================================================
# LOAD PRODUCTS
# ============================================================

DATA_FILE = os.path.join(
    "data",
    "products.csv"
)


if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        "products.csv not found. "
        "Make sure it is inside the data folder."
    )


df = pd.read_csv(DATA_FILE)

df.columns = df.columns.str.strip()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Product Name",
    "Description",
    "Category",
    "Price",
    "Rating",
    "Image"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing column in products.csv: {column}"
        )


# ============================================================
# CLEAN DATA
# ============================================================

for column in [
    "Product Name",
    "Description",
    "Category",
    "Image"
]:

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


df["Price"] = pd.to_numeric(
    df["Price"],
    errors="coerce"
).fillna(0)


df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
).fillna(0)


# ============================================================
# MACHINE LEARNING
# ============================================================


# ------------------------------------------------------------
# 1. PRODUCT NAME MODEL
# ------------------------------------------------------------

name_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)


name_matrix = name_vectorizer.fit_transform(
    df["Product Name"]
)


name_similarity = cosine_similarity(
    name_matrix,
    name_matrix
)


# ------------------------------------------------------------
# 2. DESCRIPTION MODEL
# ------------------------------------------------------------

description_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000
)


description_matrix = description_vectorizer.fit_transform(
    df["Description"]
)


description_similarity = cosine_similarity(
    description_matrix,
    description_matrix
)


# ------------------------------------------------------------
# 3. COMBINED MODEL
# ------------------------------------------------------------

df["combined_features"] = (
    df["Product Name"]
    + " "
    + df["Product Name"]
    + " "
    + df["Description"]
    + " "
    + df["Category"]
)


combined_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000
)


combined_matrix = combined_vectorizer.fit_transform(
    df["combined_features"]
)


combined_similarity = cosine_similarity(
    combined_matrix,
    combined_matrix
)


# ============================================================
# PRODUCT HELPER
# ============================================================

def product_to_dict(product):

    return {

        "name": str(
            product["Product Name"]
        ),

        "description": str(
            product["Description"]
        ),

        "category": str(
            product["Category"]
        ),

        "price": float(
            product["Price"]
        ),

        "rating": float(
            product["Rating"]
        ),

        "image": str(
            product["Image"]
        )

    }


# ============================================================
# FIND PRODUCT
# ============================================================

def find_product(product_name):

    if not product_name:

        return None


    search_name = (
        product_name
        .strip()
        .lower()
    )


    product_names = (
        df["Product Name"]
        .astype(str)
    )


    # --------------------------------------------------------
    # EXACT MATCH
    # --------------------------------------------------------

    exact = product_names[
        product_names.str.lower()
        == search_name
    ]


    if not exact.empty:

        return exact.index[0]


    # --------------------------------------------------------
    # PARTIAL MATCH
    # --------------------------------------------------------

    partial = product_names[
        product_names
        .str.lower()
        .str.contains(
            search_name,
            na=False,
            regex=False
        )
    ]


    if not partial.empty:

        return partial.index[0]


    # --------------------------------------------------------
    # FUZZY MATCH
    # --------------------------------------------------------

    names_lower = [

        name.lower()

        for name in product_names.tolist()

    ]


    close = get_close_matches(

        search_name,

        names_lower,

        n=1,

        cutoff=0.50

    )


    if close:

        return names_lower.index(
            close[0]
        )


    return None


# ============================================================
# ADVANCED RECOMMENDATION ENGINE
# ============================================================

def recommend_products(
    product_name,
    limit=6
):

    product_index = find_product(
        product_name
    )


    if product_index is None:

        return []


    selected_product = df.iloc[
        product_index
    ]


    selected_category = (
        str(
            selected_product["Category"]
        )
        .strip()
        .lower()
    )


    selected_rating = float(
        selected_product["Rating"]
    )


    scores = []


    # --------------------------------------------------------
    # COMPARE WITH EVERY PRODUCT
    # --------------------------------------------------------

    for index in range(len(df)):


        # Never recommend the same product
        if index == product_index:

            continue


        candidate = df.iloc[index]


        # ----------------------------------------------------
        # NAME SIMILARITY
        # ----------------------------------------------------

        name_score = float(

            name_similarity[
                product_index
            ][index]

        )


        # ----------------------------------------------------
        # DESCRIPTION SIMILARITY
        # ----------------------------------------------------

        description_score = float(

            description_similarity[
                product_index
            ][index]

        )


        # ----------------------------------------------------
        # COMBINED SIMILARITY
        # ----------------------------------------------------

        combined_score = float(

            combined_similarity[
                product_index
            ][index]

        )


        # ----------------------------------------------------
        # CATEGORY MATCH
        # ----------------------------------------------------

        candidate_category = (

            str(
                candidate["Category"]
            )
            .strip()
            .lower()

        )


        if candidate_category == selected_category:

            category_score = 1.0

        else:

            category_score = 0.0


        # ----------------------------------------------------
        # RATING SCORE
        # ----------------------------------------------------

        candidate_rating = float(
            candidate["Rating"]
        )


        rating_score = (
            candidate_rating / 5.0
        )


        # ----------------------------------------------------
        # RATING SIMILARITY
        # ----------------------------------------------------

        rating_difference = abs(

            selected_rating
            - candidate_rating

        )


        rating_similarity_score = max(

            0.0,

            1.0 - (
                rating_difference / 5.0
            )

        )


        # ----------------------------------------------------
        # FINAL WEIGHTED AI SCORE
        # ----------------------------------------------------

        final_score = (

            (name_score * 0.20)

            + (description_score * 0.30)

            + (combined_score * 0.25)

            + (category_score * 0.20)

            + (rating_similarity_score * 0.03)

            + (rating_score * 0.02)

        )


        scores.append({

            "index": index,

            "score": final_score,

            "name_score": name_score,

            "description_score": description_score,

            "combined_score": combined_score,

            "category_score": category_score,

            "rating_score": rating_score

        })


    # --------------------------------------------------------
    # SORT BEST RECOMMENDATIONS FIRST
    # --------------------------------------------------------

    scores.sort(

        key=lambda item:
        item["score"],

        reverse=True

    )


    recommendations = []


    # --------------------------------------------------------
    # CREATE FINAL RESULTS
    # --------------------------------------------------------

    for item in scores[:limit]:


        product = df.iloc[
            item["index"]
        ]


        recommendation = product_to_dict(
            product
        )


        # AI similarity percentage
        recommendation["similarity"] = round(

            item["score"] * 100,

            2

        )


        recommendation["category_match"] = (

            item["category_score"] == 1.0

        )


        recommendations.append(
            recommendation
        )


    return recommendations


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)

def home():

    searched_product = ""

    matched_product = None

    recommendations = []

    message = ""


    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    if request.method == "GET":


        searched_product = request.args.get(

            "product",

            ""

        ).strip()


        if not searched_product:

            searched_product = request.args.get(

                "q",

                ""

            ).strip()


    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    elif request.method == "POST":


        searched_product = request.form.get(

            "product",

            ""

        ).strip()


    # --------------------------------------------------------
    # PROCESS SEARCH
    # --------------------------------------------------------

    if searched_product:


        product_index = find_product(

            searched_product

        )


        if product_index is None:


            message = (

                "Product not found. "
                "Try another product name."

            )


        else:


            matched_product = product_to_dict(

                df.iloc[
                    product_index
                ]

            )


            recommendations = recommend_products(

                searched_product,

                limit=6

            )


            if not recommendations:

                message = (

                    "Product found, but no similar "
                    "products were found."

                )


    # --------------------------------------------------------
    # HOME PRODUCTS
    # --------------------------------------------------------

    home_products = [

        product_to_dict(row)

        for _, row in df.head(8).iterrows()

    ]


    # --------------------------------------------------------
    # CATEGORIES
    # --------------------------------------------------------

    category_list = sorted(

        df["Category"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()

    )


    return render_template(

        "index.html",

        searched_product=searched_product,

        matched_product=matched_product,

        recommendations=recommendations,

        message=message,

        products=home_products,

        categories=category_list

    )


# ============================================================
# PRODUCTS PAGE
# ============================================================

@app.route("/products")

def products():

    products_list = [

        product_to_dict(row)

        for _, row in df.iterrows()

    ]


    return render_template(

        "products.html",

        products=products_list

    )


# ============================================================
# PRODUCT DETAILS
# ============================================================

@app.route(
    "/product/<path:product_name>"
)

def product_details(product_name):

    product_index = find_product(

        product_name

    )


    if product_index is None:

        abort(404)


    product = product_to_dict(

        df.iloc[
            product_index
        ]

    )


    recommendations = recommend_products(

        product_name,

        limit=4

    )


    return render_template(

        "product_details.html",

        product=product,

        recommendations=recommendations

    )


# ============================================================
# CATEGORIES
# ============================================================

@app.route("/categories")

def categories():

    category_counts = (

        df["Category"]
        .value_counts()
        .sort_index()
        .to_dict()

    )


    categories_list = sorted(

        category_counts.keys()

    )


    return render_template(

        "categories.html",

        categories=categories_list,

        category_counts=category_counts

    )


# ============================================================
# CATEGORY PAGE
# ============================================================

@app.route(
    "/category/<path:category_name>"
)

def category(category_name):

    category_search = (

        category_name
        .strip()
        .lower()

    )


    category_products = df[

        df["Category"]
        .astype(str)
        .str.strip()
        .str.lower()

        == category_search

    ]


    if category_products.empty:

        abort(404)


    products_list = [

        product_to_dict(row)

        for _, row in category_products.iterrows()

    ]


    actual_category = str(

        category_products.iloc[0][
            "Category"
        ]

    )


    all_categories = sorted(

        df["Category"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()

    )


    return render_template(

        "category.html",

        category=actual_category,

        products=products_list,

        categories=all_categories

    )


# ============================================================
# BUY NOW
# ============================================================

@app.route(
    "/buy/<path:product_name>"
)

def buy_now(product_name):

    product_index = find_product(

        product_name

    )


    if product_index is None:

        abort(404)


    product = product_to_dict(

        df.iloc[
            product_index
        ]

    )


    return render_template(

        "buy.html",

        product=product

    )


# ============================================================
# SEARCH PAGE
# ============================================================

@app.route("/search")

def search():

    query = request.args.get(

        "q",

        ""

    ).strip()


    results = []


    if query:


        query_lower = query.lower()


        name_match = (

            df["Product Name"]
            .str.lower()
            .str.contains(

                query_lower,

                na=False,

                regex=False

            )

        )


        description_match = (

            df["Description"]
            .str.lower()
            .str.contains(

                query_lower,

                na=False,

                regex=False

            )

        )


        category_match = (

            df["Category"]
            .str.lower()
            .str.contains(

                query_lower,

                na=False,

                regex=False

            )

        )


        matches = df[

            name_match
            | description_match
            | category_match

        ]


        results = [

            product_to_dict(row)

            for _, row in matches.iterrows()

        ]


    return render_template(

        "search.html",

        query=query,

        results=results

    )


# ============================================================
# API - ALL PRODUCTS
# ============================================================

@app.route("/api/products")

def api_products():

    products_list = [

        product_to_dict(row)

        for _, row in df.iterrows()

    ]


    return jsonify({

        "success": True,

        "count": len(products_list),

        "products": products_list

    })


# ============================================================
# API - SEARCH
# ============================================================

@app.route("/api/search")

def api_search():

    query = request.args.get(

        "q",

        ""

    ).strip()


    if not query:

        return jsonify({

            "success": False,

            "count": 0,

            "products": []

        })


    query_lower = query.lower()


    matches = df[

        df["Product Name"]
        .str.lower()
        .str.contains(

            query_lower,

            na=False,

            regex=False

        )

        |

        df["Description"]
        .str.lower()
        .str.contains(

            query_lower,

            na=False,

            regex=False

        )

        |

        df["Category"]
        .str.lower()
        .str.contains(

            query_lower,

            na=False,

            regex=False

        )

    ]


    results = [

        product_to_dict(row)

        for _, row in matches.iterrows()

    ]


    return jsonify({

        "success": True,

        "count": len(results),

        "products": results

    })


# ============================================================
# API - AUTOCOMPLETE
# ============================================================

@app.route("/api/autocomplete")

def autocomplete():

    query = request.args.get(

        "q",

        ""

    ).strip().lower()


    if not query:

        return jsonify({

            "success": True,

            "suggestions": []

        })


    matches = df[

        df["Product Name"]
        .str.lower()
        .str.contains(

            query,

            na=False,

            regex=False

        )

    ]


    suggestions = []


    for _, product in matches.head(8).iterrows():

        suggestions.append({

            "name": str(
                product["Product Name"]
            ),

            "category": str(
                product["Category"]
            ),

            "price": float(
                product["Price"]
            ),

            "image": str(
                product["Image"]
            )

        })


    return jsonify({

        "success": True,

        "suggestions": suggestions

    })


# ============================================================
# API - RECOMMENDATIONS
# ============================================================

@app.route(
    "/api/recommend/<path:product_name>"
)

def api_recommend(product_name):

    product_index = find_product(

        product_name

    )


    if product_index is None:

        return jsonify({

            "success": False,

            "message": "Product not found",

            "recommendations": []

        }), 404


    recommendations = recommend_products(

        product_name,

        limit=6

    )


    return jsonify({

        "success": True,

        "product": product_name,

        "recommendations": recommendations

    })


# ============================================================
# SHOPPING CART HELPERS
# ============================================================

def get_cart():

    return session.get(
        "cart",
        {}
    )


def save_cart(cart):

    session["cart"] = cart

    session.modified = True


# ============================================================
# ADD TO CART
# ============================================================

@app.route(
    "/cart/add/<path:product_name>"
)

def add_to_cart(product_name):

    product_index = find_product(

        product_name

    )


    if product_index is None:

        abort(404)


    cart = get_cart()


    if product_name in cart:

        cart[product_name] += 1

    else:

        cart[product_name] = 1


    save_cart(cart)


    return redirect(

        request.referrer
        or url_for("home")

    )


# ============================================================
# CART PAGE
# ============================================================

@app.route("/cart")

def cart():

    cart = get_cart()

    cart_items = []

    subtotal = 0


    for product_name, quantity in cart.items():


        product_index = find_product(

            product_name

        )


        if product_index is None:

            continue


        product = product_to_dict(

            df.iloc[
                product_index
            ]

        )


        quantity = int(quantity)


        product["quantity"] = quantity


        product["item_total"] = round(

            product["price"] * quantity,

            2

        )


        subtotal += product["item_total"]


        cart_items.append(product)


    return render_template(

        "cart.html",

        cart_items=cart_items,

        subtotal=round(
            subtotal,
            2
        ),

        cart_count=sum(
            cart.values()
        )

    )


# ============================================================
# REMOVE FROM CART
# ============================================================

@app.route(
    "/cart/remove/<path:product_name>"
)

def remove_from_cart(product_name):

    cart = get_cart()


    if product_name in cart:

        del cart[product_name]


    save_cart(cart)


    return redirect(

        url_for("cart")

    )


# ============================================================
# INCREASE CART QUANTITY
# ============================================================

@app.route(
    "/cart/increase/<path:product_name>"
)

def increase_cart_quantity(product_name):

    cart = get_cart()


    if product_name in cart:

        cart[product_name] += 1


    save_cart(cart)


    return redirect(

        url_for("cart")

    )


# ============================================================
# DECREASE CART QUANTITY
# ============================================================

@app.route(
    "/cart/decrease/<path:product_name>"
)

def decrease_cart_quantity(product_name):

    cart = get_cart()


    if product_name in cart:

        cart[product_name] -= 1


        if cart[product_name] <= 0:

            del cart[product_name]


    save_cart(cart)


    return redirect(

        url_for("cart")

    )


# ============================================================
# CLEAR CART
# ============================================================

@app.route("/cart/clear")

def clear_cart():

    session["cart"] = {}


    return redirect(

        url_for("cart")

    )


# ============================================================
# CHECKOUT
# ============================================================

@app.route(
    "/checkout",
    methods=["GET", "POST"]
)

def checkout():

    cart = get_cart()


    # Cart must contain products
    if not cart:

        return redirect(

            url_for("products")

        )


    cart_items = []

    subtotal = 0


    # --------------------------------------------------------
    # BUILD CART ITEMS
    # --------------------------------------------------------

    for product_name, quantity in cart.items():


        product_index = find_product(

            product_name

        )


        if product_index is None:

            continue


        product = product_to_dict(

            df.iloc[
                product_index
            ]

        )


        quantity = int(quantity)


        product["quantity"] = quantity


        product["item_total"] = round(

            product["price"] * quantity,

            2

        )


        subtotal += product["item_total"]


        cart_items.append(product)


    subtotal = round(
        subtotal,
        2
    )


    # --------------------------------------------------------
    # INVALID CART
    # --------------------------------------------------------

    if not cart_items:

        session["cart"] = {}


        return redirect(

            url_for("products")

        )


    # --------------------------------------------------------
    # PLACE ORDER
    # --------------------------------------------------------

    if request.method == "POST":


        customer_name = request.form.get(

            "customer_name",

            ""

        ).strip()


        phone = request.form.get(

            "phone",

            ""

        ).strip()


        address = request.form.get(

            "address",

            ""

        ).strip()


        city = request.form.get(

            "city",

            ""

        ).strip()


        pincode = request.form.get(

            "pincode",

            ""

        ).strip()


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not customer_name:

            return render_template(

                "checkout.html",

                cart_items=cart_items,

                subtotal=subtotal,

                error="Please enter your name."

            )


        if not phone:

            return render_template(

                "checkout.html",

                cart_items=cart_items,

                subtotal=subtotal,

                error="Please enter your phone number."

            )


        if not address:

            return render_template(

                "checkout.html",

                cart_items=cart_items,

                subtotal=subtotal,

                error="Please enter your address."

            )


        if not city:

            return render_template(

                "checkout.html",

                cart_items=cart_items,

                subtotal=subtotal,

                error="Please enter your city."

            )


        if not pincode:

            return render_template(

                "checkout.html",

                cart_items=cart_items,

                subtotal=subtotal,

                error="Please enter your pincode."

            )


        # ----------------------------------------------------
        # CREATE ORDER
        # ----------------------------------------------------

        order = {

            "customer_name":
                customer_name,

            "phone":
                phone,

            "address":
                address,

            "city":
                city,

            "pincode":
                pincode,

            "items":
                cart_items,

            "total":
                subtotal

        }


        # Save order
        session["last_order"] = order


        # Clear cart
        session["cart"] = {}


        session.modified = True


        # Go to success page
        return redirect(

            url_for(
                "order_success"
            )

        )


    # --------------------------------------------------------
    # SHOW CHECKOUT PAGE
    # --------------------------------------------------------

    return render_template(

        "checkout.html",

        cart_items=cart_items,

        subtotal=subtotal,

        error=""

    )


# ============================================================
# ORDER SUCCESS
# ============================================================

@app.route("/order-success")

def order_success():

    order = session.get(
        "last_order"
    )


    if not order:

        return redirect(

            url_for("home")

        )


    return render_template(

        "order_success.html",

        order=order

    )


# ============================================================
# 404 ERROR
# ============================================================

@app.errorhandler(404)

def page_not_found(error):

    return render_template(
        "404.html"
    ), 404


# ============================================================
# RUN FLASK APP
# ============================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )