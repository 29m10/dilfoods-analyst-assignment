***SQL Submissions***

Write a query to get the total number of customers who signed up in 2023

```
select count(customer_id) total_customers from customers where date_format(signup_date, '%y') = 2023;
```

Write a query to calculate total revenue for each month in 2023, grouped by month

```
select month(p.payment_date), sum(p.amount) total_revenue from orders o
inner join payments p on p.order_id = o.order_id
where p.status = 'paid' and year(p.payment_date) = 2023
group by 1
order by 1
```

Find the top 3 most sold products based on order quantities

```
select 
    p.product_name, 
    sum(oi.quantity) total_sold
from order_items oi
join products p on oi.product_id = p.product_id
group by 1
order by 1 desc
limit 3;
```

Retrieve a list of customers and their total spend. Show only those who
spent more than $500

```
select 
    c.customer_id, 
    c.name, 
    SUM(o.total_amount) total_spent
from customers c
join orders o on c.customer_id = o.customer_id
group by c.customer_id, c.name
having total_spent > 500
order by 3 desc;
```

List all products with stock quantity below 50 (low stock alert)

```
select product_name, stock_quantity
from products
where stock_quantity < 50;
```

Identify all orders that were completed but have no matching payment
records (possible refunds)

```
select * from orders o
where o.order_id in (
    select order_id from payments where status != 'completed'
) 
and o.status = 'completed';

```